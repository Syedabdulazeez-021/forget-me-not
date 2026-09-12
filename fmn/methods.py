"""Continual-learning strategies.

Every strategy is a subclass of `Strategy` and overrides at most three hooks:

    before_task(...)   -- set up anything the task needs (freezing, buffers)
    extra_loss(...)    -- a penalty added to the cross-entropy
    after_task(...)    -- consolidate (store Fisher, snapshot teacher, ...)

That uniformity is deliberate: it means the *only* thing differing between a
baseline row and a method row in the results table is the strategy object, so
no accidental difference in optimiser, schedule, or data order can be
mistaken for a method effect.

Implemented:
    naive         -- sequential fine-tuning (mandatory lower bound)
    joint         -- train on all tasks at once (mandatory upper bound)
    replay        -- experience replay with a reservoir buffer
    ewc           -- Elastic Weight Consolidation (Kirkpatrick et al., 2017)
    lwf           -- Learning without Forgetting (Li & Hoiem, 2016)
    drift_freeze  -- ours: freeze the layers a naive run showed to drift most
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .selection import (load_param_counts, load_profile, matched_param_budget,
                        parse_explicit_layers, rank_layers_by_drift, select_layers)


# --------------------------------------------------------------------------
# reservoir buffer
# --------------------------------------------------------------------------
class ReservoirBuffer:
    """Fixed-capacity buffer filled by reservoir sampling (Vitter, 1985).

    Reservoir sampling keeps a uniform sample of the whole stream seen so far
    without knowing its length in advance -- which is exactly the continual
    setting. It also means the buffer is not biased toward the most recent
    task, which a naive ring buffer would be.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.t: Optional[torch.Tensor] = None
        self.n_seen = 0
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(1234)
        self._np_rng = np.random.RandomState(1234)

    def __len__(self) -> int:
        return 0 if self.x is None else self.x.shape[0]

    def _init(self, x: torch.Tensor, y: torch.Tensor):
        shape = (self.capacity,) + tuple(x.shape[1:])
        self.x = torch.zeros(shape, dtype=x.dtype, device=self.device)
        self.y = torch.zeros(self.capacity, dtype=y.dtype, device=self.device)
        self.t = torch.zeros(self.capacity, dtype=torch.long, device=self.device)

    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        """Vectorised reservoir insert.

        Equivalent to the textbook per-sample loop but done with one scatter,
        because the loop version costs a Python iteration per training sample
        and dominated runtime on anything larger than MNIST.
        """
        if self.capacity <= 0:
            return
        if self.x is None:
            self._init(x, y)

        b = int(x.shape[0])
        stream_pos = self.n_seen + np.arange(b)            # index of each sample in the stream
        dest = np.full(b, -1, dtype=np.int64)

        fill = stream_pos < self.capacity                  # buffer not yet full
        dest[fill] = stream_pos[fill]

        rest = ~fill
        if rest.any():
            j = self._np_rng.randint(0, stream_pos[rest] + 1)
            dest[rest] = np.where(j < self.capacity, j, -1)

        sel = np.where(dest >= 0)[0]
        if sel.size:
            # two samples in one batch can target the same slot; keep the last
            # write so the result matches the sequential loop exactly and does
            # not depend on scatter ordering
            d = dest[sel]
            _, first_in_reversed = np.unique(d[::-1], return_index=True)
            keep = sel.size - 1 - first_in_reversed
            keep.sort()
            sel, d = sel[keep], dest[sel[keep]]

            idx = torch.from_numpy(d).to(self.device)
            src = torch.from_numpy(sel)
            self.x[idx] = x[src].to(self.device)
            self.y[idx] = y[src].to(self.device)
            self.t[idx] = task_id

        self.n_seen += b

    def sample(self, n: int):
        size = len(self)
        if size == 0:
            return None
        n = min(n, size)
        idx = torch.randint(0, size, (n,), generator=self._gen).to(self.device)
        return self.x[idx], self.y[idx], self.t[idx]


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class Strategy:
    name = "base"
    #: joint training needs the whole stream at once; the trainer checks this
    trains_jointly = False

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device

    def before_task(self, task_index: int, task, seq) -> None:
        pass

    def observe(self, x, y, task_index: int):
        """Hook for strategies that need to modify the batch itself.

        Returns (x, y) actually fed to the network. Default: unchanged.
        """
        return x, y

    def extra_loss(self, x, y, logits, task_index: int) -> torch.Tensor:
        return torch.zeros((), device=self.device)

    def after_step(self, x, y, task_index: int) -> None:
        pass

    def before_optimizer_step(self) -> None:
        """Called after gradients are unscaled, before clipping and stepping."""
        pass

    def on_train_mode(self) -> None:
        """Called after every `model.train()`.

        `model.train()` re-enables BatchNorm updates everywhere, so any
        strategy that deliberately disabled them has to re-assert that here or
        the setting silently evaporates at the start of each epoch.
        """
        pass

    def after_task(self, task_index: int, task, seq, loader_fn, mask_fn) -> None:
        """`mask_fn(logits, y)` applies the same logit masking used in training."""
        pass


class Naive(Strategy):
    """Sequential fine-tuning. The mandatory lower bound."""

    name = "naive"


class Joint(Strategy):
    """Train once on the union of all tasks. The mandatory upper bound."""

    name = "joint"
    trains_jointly = True


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
class Replay(Strategy):
    """Experience replay: concatenate buffer samples onto every batch."""

    name = "replay"

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.buffer = ReservoirBuffer(cfg.buffer_size, device)
        self._current_task = 0

    def before_task(self, task_index, task, seq):
        self._current_task = task_index

    def observe(self, x, y, task_index):
        if task_index == 0 or len(self.buffer) == 0 or self.cfg.replay_batch <= 0:
            return x, y
        got = self.buffer.sample(self.cfg.replay_batch)
        if got is None:
            return x, y
        bx, by, _ = got
        return torch.cat([x, bx], dim=0), torch.cat([y, by], dim=0)

    def after_step(self, x, y, task_index):
        # store the *original* batch, not the replay-augmented one, or the
        # buffer would resample its own contents and collapse in diversity
        self.buffer.add(x.detach(), y.detach(), task_index)


# --------------------------------------------------------------------------
# EWC
# --------------------------------------------------------------------------
class EWC(Strategy):
    """Elastic Weight Consolidation with a diagonal empirical Fisher.

    After each task we estimate, for every parameter, how sensitive the
    task's log-likelihood was to it, and then penalise moving the parameters
    that mattered. We keep one (Fisher, anchor) pair per completed task and
    sum their penalties, which is the original formulation; with 5 tasks the
    memory cost is trivial and it avoids the drift that online-EWC's single
    running Fisher introduces.
    """

    name = "ewc"

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.anchors: List[Dict[str, torch.Tensor]] = []
        self.fishers: List[Dict[str, torch.Tensor]] = []

    def extra_loss(self, x, y, logits, task_index):
        if not self.fishers:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        params = dict(self.model.named_parameters())
        for fisher, anchor in zip(self.fishers, self.anchors):
            for n, p in params.items():
                if n in fisher:
                    penalty = penalty + (fisher[n] * (p - anchor[n]).pow(2)).sum()
        return 0.5 * self.cfg.ewc_lambda * penalty

    @torch.no_grad()
    def _store_anchor(self):
        self.anchors.append(
            {n: p.detach().clone() for n, p in self.model.named_parameters()}
        )

    def after_task(self, task_index, task, seq, loader_fn, mask_fn):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        loader = loader_fn(task.train, shuffle=True)
        self.model.eval()
        n_batches = 0
        for bi, (x, y) in enumerate(loader):
            if bi >= self.cfg.ewc_fisher_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            # Fisher must be estimated under the SAME output distribution the
            # model was trained with; skipping the mask here would measure
            # sensitivity of a distribution the model never optimised.
            logits = mask_fn(self.model(x).float(), y)
            # empirical Fisher: gradient of the log-likelihood of the TRUE label
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            n_batches += 1
        if n_batches:
            for n in fisher:
                fisher[n] /= n_batches
        self.model.zero_grad(set_to_none=True)
        self.fishers.append(fisher)
        self._store_anchor()


# --------------------------------------------------------------------------
# LwF
# --------------------------------------------------------------------------
class LwF(Strategy):
    """Learning without Forgetting: distil the previous model on new data."""

    name = "lwf"

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.teacher: Optional[nn.Module] = None
        self.seen_classes: List[int] = []

    def extra_loss(self, x, y, logits, task_index):
        if self.teacher is None or not self.seen_classes:
            return torch.zeros((), device=self.device)
        T = self.cfg.lwf_temperature
        idx = torch.tensor(self.seen_classes, device=self.device, dtype=torch.long)
        with torch.no_grad():
            t_logits = self.teacher(x)[:, idx]
        s_logits = logits[:, idx]
        kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        return self.cfg.lwf_alpha * kd

    def after_task(self, task_index, task, seq, loader_fn, mask_fn):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for c in task.classes:
            if c not in self.seen_classes:
                self.seen_classes.append(c)


# --------------------------------------------------------------------------
# ours
# --------------------------------------------------------------------------
class DriftFreeze(Strategy):
    """Freeze the layer groups a prior naive run showed to drift most.

    Two-stage by construction:
      1. run `--method naive`, which writes `drift.json`
      2. run `--method drift_freeze --drift-profile .../drift.json --freeze-topk K`

    Freezing starts at task 1: task 0 must train the whole network or there
    is nothing to preserve. The head is never frozen (see drift.rank_layers_by_drift).

    `--freeze-topk 0` is a deliberate no-op that reduces exactly to naive; it
    is the k=0 point of the ablation curve and a useful correctness check.
    """

    name = "drift_freeze"

    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        if not cfg.drift_profile:
            raise ValueError(
                "drift_freeze needs --drift-profile pointing at a naive run's drift.json"
            )
        self.profile = load_profile(cfg.drift_profile)
        self.param_counts = load_param_counts(cfg.drift_profile)
        self.ranked = rank_layers_by_drift(self.profile)
        if cfg.freeze_layers:
            # explicit list: used by the forgetting-risk scan, which isolates
            # one layer at a time rather than sweeping a ranked budget
            self.frozen: List[str] = parse_explicit_layers(
                cfg.freeze_layers, self.profile)
            self.select_mode = "explicit"
        else:
            self.frozen = select_layers(
                self.profile, cfg.freeze_topk, mode=cfg.freeze_select,
                param_counts=self.param_counts, seed=cfg.seed,
            )
            self.select_mode = cfg.freeze_select
        self.groups = model.layer_groups()
        self.weight_masks = None
        self.budget = matched_param_budget(
            self.profile, cfg.freeze_topk, self.param_counts)
        if self.select_mode == "paramrandom" and cfg.freeze_topk > 0:
            self.weight_masks = self._build_weight_masks(self.budget)

    def _build_weight_masks(self, budget: int):
        """Randomly pick exactly `budget` individual weights to hold fixed.

        Candidates exclude the classifier head, matching the layer-level arms.
        The mask is drawn once and reused for every task: freezing a different
        random set each task would be a weaker and different experiment.
        """
        head_names = set(self.groups.get("head", []))
        named = [(n, p) for n, p in self.model.named_parameters()
                 if n not in head_names]
        sizes = [p.numel() for _, p in named]
        total = sum(sizes)
        budget = min(budget, total)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.seed)
        flat = torch.zeros(total, dtype=torch.bool)
        if budget > 0:
            flat[torch.randperm(total, generator=g)[:budget]] = True
        masks, off = {}, 0
        for (n, prm), sz in zip(named, sizes):
            masks[n] = flat[off:off + sz].view_as(prm).to(prm.device)
            off += sz
        return masks

    def before_optimizer_step(self) -> None:
        """Zero the gradients of masked weights, after unscaling.

        Individual weights cannot be frozen via `requires_grad`, which is
        per-tensor. Zeroing their gradients immediately before the optimiser
        step is the equivalent operation. Every arm uses the same optimiser,
        so any residual momentum effect applies identically across arms and
        cannot explain a difference between them.
        """
        if self.weight_masks is None:
            return
        for n, prm in self.model.named_parameters():
            m = self.weight_masks.get(n)
            if m is not None and prm.grad is not None:
                prm.grad.masked_fill_(m, 0.0)

    def _param_names_to_freeze(self) -> List[str]:
        names: List[str] = []
        for g in self.frozen:
            names.extend(self.groups.get(g, []))
        return names

    def before_task(self, task_index, task, seq):
        self._active = task_index >= 1 and bool(self.frozen)
        targets = set(self._param_names_to_freeze()) if self._active else set()
        self._frozen_params = targets
        for n, p in self.model.named_parameters():
            p.requires_grad_(n not in targets)
        self.on_train_mode()

    def on_train_mode(self) -> None:
        """Put fully-frozen BatchNorm modules back into eval mode.

        Freezing a BatchNorm's weight and bias is not enough: its running mean
        and variance are buffers, not parameters, and keep absorbing new-task
        statistics regardless of requires_grad. That leaks the new task into
        old-task predictions and would make a frozen layer look like it still
        forgets.

        Resolved by parameter ownership rather than by attribute name, because
        group names like "stem" and "layer1" do not correspond one-to-one with
        module attributes across backbones -- matching on names silently missed
        the stem BatchNorm in ReducedResNet18.
        """
        if not getattr(self, "_active", False):
            return
        frozen = getattr(self, "_frozen_params", set())
        for mod_name, m in self.model.named_modules():
            if not isinstance(m, nn.modules.batchnorm._BatchNorm):
                continue
            pnames = [f"{mod_name}.{pn}" if mod_name else pn
                      for pn, _ in m.named_parameters(recurse=False)]
            if pnames and all(pn in frozen for pn in pnames):
                m.eval()

    def info(self) -> dict:
        return {
            "select_mode": self.select_mode,
            "freeze_topk": self.cfg.freeze_topk,
            "ranked_layers_by_drift": self.ranked,
            "frozen_layers": self.frozen,
            # reported so the results table can show that a control arm froze
            # a comparable number of weights, not just a comparable number of
            # layers -- the confound the matched arm exists to rule out
            "frozen_param_count": int(
                self.budget if self.weight_masks is not None
                else sum(self.param_counts.get(g, 0) for g in self.frozen)),
            "freeze_granularity": "weights" if self.weight_masks is not None else "layers",
            "drift_profile": self.profile,
        }


REGISTRY = {
    "naive": Naive,
    "joint": Joint,
    "replay": Replay,
    "ewc": EWC,
    "lwf": LwF,
    "drift_freeze": DriftFreeze,
}


def build_strategy(name: str, model, cfg, device) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](model, cfg, device)
