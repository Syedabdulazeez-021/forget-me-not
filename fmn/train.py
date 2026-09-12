"""Continual-learning training loop.

One command trains one (dataset, scenario, method, seed) combination end to
end and writes everything needed to reproduce and report it:

    runs/<name>/config.json      exact configuration, including the seed
    runs/<name>/acc_matrix.csv   R[i][j] = acc on task j after task i
    runs/<name>/results.json     headline metrics + per-task detail
    runs/<name>/drift.json       per-layer drift profile
    runs/<name>/log.txt          human-readable trace

Usage:
    python -m fmn.train --dataset split-mnist --method naive
    python -m fmn.train --dataset split-mnist --method replay --buffer-size 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RunConfig, get_device, seed_everything
from .data import build_task_sequence, concat_train_subsets, make_loader
from .drift import DriftTracker
from .methods import build_strategy
from .metrics import summarize
from .models import build_model, count_parameters

NEG_INF = -1e9  # finite, so it never produces NaN in softmax/cross-entropy


class Tee:
    """Write to stdout and a log file at once."""

    def __init__(self, path: str):
        # encoding pinned: Windows defaults to cp1252, which cannot encode
        # several characters these files legitimately contain
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def additive_mask(allowed: List[int], n_classes: int, device, dtype=torch.float32):
    """Row vector that is 0 on `allowed` classes and very negative elsewhere."""
    m = torch.full((n_classes,), NEG_INF, device=device, dtype=dtype)
    m[torch.tensor(sorted(allowed), device=device, dtype=torch.long)] = 0.0
    return m


def mask_logits(logits: torch.Tensor, allowed: List[int]) -> torch.Tensor:
    """Mask every row of `logits` with the same allowed-class set."""
    return logits + additive_mask(allowed, logits.shape[1], logits.device, logits.dtype)


class LogitMasker:
    """Per-sample logit masking.

    A batch is not always single-task. Experience replay splices in samples
    from earlier tasks, and joint training draws from every task at once. If
    the whole batch is masked with the *current* task's classes, the replayed
    samples have their own correct label forced to a near-infinite negative
    logit, so cross-entropy on them is enormous and their gradient is garbage.
    That silently turns replay from the strongest baseline into a broken one.

    Since each class belongs to exactly one task, a sample's label recovers its
    task, so every row can be masked with the classes of the task it actually
    came from.

    In class-incremental mode there is no per-task restriction: every row is
    masked with the union of classes seen so far.
    """

    def __init__(self, seq, scenario: str, device):
        self.scenario = scenario
        self.device = device
        self.n_classes = seq.n_classes
        c2t = seq.class_to_task
        assigned = [c for t in seq for c in t.classes]
        if any(c2t[c] < 0 for c in assigned):
            raise RuntimeError("class_to_task is missing an assigned class")
        # unassigned ids (possible when n_classes % classes_per_task != 0) are
        # clamped to task 0 so indexing can never wrap to the last row
        self.class_to_task = torch.from_numpy(c2t.clip(min=0)).to(device)
        M = seq.task_class_mask_matrix().to(device)
        # [n_tasks, n_classes] additive mask, one row per task
        self.task_rows = torch.where(
            M, torch.zeros_like(M, dtype=torch.float32),
            torch.full_like(M, NEG_INF, dtype=torch.float32),
        )
        self.seen_row = None

    def set_seen(self, seen_classes: List[int]) -> None:
        self.seen_row = additive_mask(seen_classes, self.n_classes, self.device)

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.scenario == "class":
            return logits + self.seen_row
        return logits + self.task_rows[self.class_to_task[y]]


@torch.no_grad()
def evaluate(model, loader, allowed: List[int], device, amp: bool) -> float:
    model.eval()
    correct = total = 0
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        logits = mask_logits(logits.float(), allowed)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def make_grad_scaler(device, enabled: bool):
    """GradScaler across torch versions.

    torch >= 2.4 wants `torch.amp.GradScaler(device_type, ...)`; older builds
    only have `torch.cuda.amp.GradScaler(...)`. A disabled scaler is a no-op
    on every path (unscale_ and step both return early), so CPU runs are safe.
    """
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def run(cfg: RunConfig) -> dict:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    out_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    log = Tee(os.path.join(out_dir, "log.txt"))

    log(f"[cfg] {json.dumps(cfg.to_dict())}")
    log(f"[env] torch={torch.__version__} device={device}")

    seq = build_task_sequence(cfg.dataset, data_root=cfg.data_root)
    n_tasks = len(seq)
    log(f"[data] {cfg.dataset}: {n_tasks} tasks, class order {seq.class_order}")
    for t in seq:
        log(f"       {t}")

    model = build_model(seq.spec, seq.n_classes).to(device)
    log(f"[model] {type(model).__name__} params={count_parameters(model):,}")

    strategy = build_strategy(cfg.method, model, cfg, device)
    log(f"[method] {strategy.name}")
    if hasattr(strategy, "info"):
        log(f"[method] {json.dumps(strategy.info())}")

    tracker = DriftTracker(model)

    def loader_fn(subset, shuffle=True):
        return make_loader(subset, cfg.batch_size, shuffle,
                           num_workers=cfg.num_workers, seed=cfg.seed)

    eval_split = (lambda task: task.test) if cfg.eval_on == "test" else (lambda task: task.val)
    # persistent=False: these loaders live for the whole run, and keeping
    # n_tasks x num_workers processes alive for that long is a real resource leak
    eval_loaders = [
        make_loader(eval_split(t), cfg.batch_size, False,
                    num_workers=cfg.num_workers, seed=cfg.seed, persistent=False)
        for t in seq
    ]

    masker = LogitMasker(seq, cfg.scenario, device)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = make_grad_scaler(device, use_amp)

    R = np.zeros((n_tasks, n_tasks), dtype=float)
    seen_classes: List[int] = []
    t_start = time.time()

    for ti, task in enumerate(seq):
        seen_classes.extend(c for c in task.classes if c not in seen_classes)
        strategy.before_task(ti, task, seq)

        if strategy.trains_jointly:
            # cumulative upper bound: at stage i the model may see every task
            # up to and including i. This gives a genuine accuracy matrix
            # rather than a single final number, at the same per-stage epoch
            # budget as the sequential methods.
            train_subset = concat_train_subsets(seq, upto=ti)
        else:
            train_subset = task.train

        train_loader = loader_fn(train_subset, shuffle=True)
        optimizer = build_optimizer(model, cfg)

        masker.set_seen(sorted(seen_classes))

        for epoch in range(cfg.epochs_per_task):
            model.train()
            strategy.on_train_mode()   # re-assert anything model.train() undid
            running, n_batches = 0.0, 0
            for bi, (x, y) in enumerate(train_loader):
                if cfg.max_batches_per_epoch and bi >= cfg.max_batches_per_epoch:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x_in, y_in = strategy.observe(x, y, ti)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x_in)
                    loss = F.cross_entropy(masker(logits, y_in), y_in)
                    loss = loss + strategy.extra_loss(x_in, y_in, logits.float(), ti)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                strategy.before_optimizer_step()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                scaler.step(optimizer)
                scaler.update()

                strategy.after_step(x, y, ti)
                running += loss.item()
                n_batches += 1

            log(f"[train] task {ti} epoch {epoch + 1}/{cfg.epochs_per_task} "
                f"loss={running / max(n_batches, 1):.4f}")

        strategy.after_task(ti, task, seq, loader_fn, masker)

        # ---- evaluate on every task, seen or not ----
        for tj in range(n_tasks):
            allowed = seq[tj].classes if cfg.scenario == "task" else sorted(seen_classes)
            R[ti, tj] = evaluate(model, eval_loaders[tj], allowed, device, cfg.amp)
        log(f"[eval ] after task {ti}: " +
            " ".join(f"T{j}={R[ti, j]:.4f}" for j in range(n_tasks)))

        # ---- drift bookkeeping ----
        if ti == 0:
            tracker.set_reference(model)
        else:
            d = tracker.record(model)
            log("[drift] " + " ".join(f"{g}={v['rel_l2']:.4f}" for g, v in d.items()))

    elapsed = time.time() - t_start
    summary = summarize(R)
    summary.update({
        "method": cfg.method,
        "dataset": cfg.dataset,
        "scenario": cfg.scenario,
        "seed": cfg.seed,
        "wall_seconds": round(elapsed, 1),
        "n_parameters": count_parameters(model),
    })
    if hasattr(strategy, "info"):
        summary["method_info"] = strategy.info()

    np.savetxt(os.path.join(out_dir, "acc_matrix.csv"), R, delimiter=",", fmt="%.6f")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    tracker.save(os.path.join(out_dir, "drift.json"))

    log("")
    log(f"[RESULT] method={cfg.method:<13} "
        f"avg_acc={summary['average_accuracy']:.4f}  "
        f"avg_forget={summary['average_forgetting']:.4f}  "
        f"bwt={summary['backward_transfer']:+.4f}  "
        f"({elapsed:.0f}s)")
    log(f"[saved ] {out_dir}")
    log.close()
    return summary


def parse_args(argv=None) -> RunConfig:
    p = argparse.ArgumentParser(description="Forget-Me-Not continual learning runner")
    d = RunConfig()
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--scenario", default=d.scenario, choices=["task", "class"])
    p.add_argument("--method", default=d.method)
    p.add_argument("--epochs-per-task", type=int, default=d.epochs_per_task)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--optimizer", default=d.optimizer, choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--buffer-size", type=int, default=d.buffer_size)
    p.add_argument("--replay-batch", type=int, default=d.replay_batch)
    p.add_argument("--ewc-lambda", type=float, default=d.ewc_lambda)
    p.add_argument("--ewc-fisher-batches", type=int, default=d.ewc_fisher_batches)
    p.add_argument("--lwf-alpha", type=float, default=d.lwf_alpha)
    p.add_argument("--lwf-temperature", type=float, default=d.lwf_temperature)
    p.add_argument("--freeze-topk", type=int, default=d.freeze_topk)
    p.add_argument("--freeze-layers", default=d.freeze_layers,
                   help="explicit comma-separated layers to freeze, e.g. 'layer3'; "
                        "overrides --freeze-select/--freeze-topk")
    p.add_argument("--freeze-select", default=d.freeze_select,
                   choices=["top", "bottom", "random", "paramrandom"],
                   help="top = the method; bottom/random/matched are controls")
    p.add_argument("--drift-profile", default=d.drift_profile)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=d.amp)
    p.add_argument("--eval-on", default=d.eval_on, choices=["test", "val"])
    p.add_argument("--max-batches-per-epoch", type=int, default=d.max_batches_per_epoch,
                   help="truncate each epoch; use for fast pipeline smoke tests only")
    a = p.parse_args(argv)
    return RunConfig(
        dataset=a.dataset, scenario=a.scenario, method=a.method,
        epochs_per_task=a.epochs_per_task, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, optimizer=a.optimizer, momentum=a.momentum,
        buffer_size=a.buffer_size, replay_batch=a.replay_batch,
        ewc_lambda=a.ewc_lambda, ewc_fisher_batches=a.ewc_fisher_batches,
        lwf_alpha=a.lwf_alpha, lwf_temperature=a.lwf_temperature,
        freeze_topk=a.freeze_topk, freeze_select=a.freeze_select,
        freeze_layers=a.freeze_layers,
        drift_profile=a.drift_profile,
        seed=a.seed, device=a.device, num_workers=a.num_workers,
        data_root=a.data_root, out_dir=a.out_dir, tag=a.tag, amp=a.amp,
        eval_on=a.eval_on, max_batches_per_epoch=a.max_batches_per_epoch,
    )


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
