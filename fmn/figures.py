"""Figure generation.

Three figures carry the entire story. Nothing else is worth the time.

  fig1_forgetting_curve.png  accuracy on Task 0 as tasks 1..T arrive, one
                             line per method. The naive line falls off a
                             cliff; yours does not. This is the demo slide.
  fig2_drift_heatmap.png     per-layer drift across checkpoints. This is the
                             diagnosis that motivates the method.
  fig3_ablation.png          average accuracy and forgetting vs the swept
                             hyper-parameter (freeze-topk or buffer size).

Usage:
    python -m fmn.figures --runs runs --out figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PRETTY = {
    "naive": "Naive fine-tuning (lower bound)",
    "joint": "Cumulative joint (upper bound)",
    "replay": "Experience replay",
    "ewc": "EWC",
    "lwf": "LwF",
    "drift_freeze": "Drift-freeze (ours)",
}
ORDER = ["naive", "lwf", "ewc", "replay", "drift_freeze", "joint"]


def load_runs(runs_dir: str) -> Dict[str, dict]:
    out = {}
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        rj, am = os.path.join(d, "results.json"), os.path.join(d, "acc_matrix.csv")
        if not (os.path.isfile(rj) and os.path.isfile(am)):
            continue
        with open(rj, encoding="utf-8") as f:
            res = json.load(f)
        res["acc_matrix"] = np.atleast_2d(np.loadtxt(am, delimiter=","))
        res["_dir"] = d
        cfg_path = os.path.join(d, "config.json")
        res["_config"] = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                res["_config"] = json.load(f)
        drift_path = os.path.join(d, "drift.json")
        if os.path.isfile(drift_path):
            with open(drift_path, encoding="utf-8") as f:
                res["drift"] = json.load(f)
        out[name] = res
    return out


def _sorted_by_method(runs: Dict[str, dict]) -> List[tuple]:
    def key(item):
        m = item[1].get("method", "")
        return (ORDER.index(m) if m in ORDER else 99, item[0])
    return sorted(runs.items(), key=key)


def _k_of(res: dict) -> int:
    """The freeze budget k for a drift_freeze run.

    Must come from the configured `freeze_topk`, not from `len(frozen_layers)`:
    the `paramrandom` arm freezes scattered weights and no whole layer group,
    so its layer list is always empty and every k would collapse onto zero.
    """
    info = res.get("method_info", {})
    if "freeze_topk" in info:
        return int(info["freeze_topk"])
    cfg = res.get("_config", {})
    if "freeze_topk" in cfg:
        return int(cfg["freeze_topk"])
    return len(info.get("frozen_layers", []))


def _label_for(name: str, res: dict) -> str:
    """Human label that distinguishes runs sharing a method.

    Without the hyper-parameter suffix, four `drift_freeze` runs at different
    freeze budgets all render as the same string, which makes the legend and
    the table unreadable.
    """
    method = res.get("method")
    label = PRETTY.get(method, name)
    cfg = res.get("_config", {})
    if method == "drift_freeze":
        info = res.get("method_info", {})
        mode = info.get("select_mode", cfg.get("freeze_select", "top"))
        label += f" [{mode}, k={_k_of(res)}]"
    elif method == "replay" and "buffer_size" in cfg:
        label += f" (buffer={cfg['buffer_size']})"
    return label


def _primary_seed(runs: Dict[str, dict]) -> int:
    seeds = [r.get("seed") for r in runs.values() if r.get("seed") is not None]
    return min(seeds) if seeds else None


def _select_for_curve(runs: Dict[str, dict]) -> List[tuple]:
    """One line per method for the headline figure.

    Plotting every seed and every ablation point puts ~18 lines on one chart,
    which communicates nothing. Keep the primary seed, and for the swept method
    keep only its best setting -- the ablation has its own figure.
    """
    seed = _primary_seed(runs)
    pool = [(n, r) for n, r in _sorted_by_method(runs)
            if seed is None or r.get("seed") == seed]
    best_df, out = None, []
    for n, r in pool:
        if r.get("method") == "drift_freeze":
            if best_df is None or r["average_accuracy"] > best_df[1]["average_accuracy"]:
                best_df = (n, r)
        else:
            out.append((n, r))
    if best_df:
        out.append(best_df)
    return sorted(out, key=lambda it: (ORDER.index(it[1].get("method"))
                                       if it[1].get("method") in ORDER else 99))


def fig_forgetting_curve(runs: Dict[str, dict], out_path: str, task: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in _select_for_curve(runs):
        R = res["acc_matrix"]
        if R.shape[0] <= task:
            continue
        xs = np.arange(task, R.shape[0])
        ys = R[task:, task]
        label = _label_for(name, res)
        style = "--" if res.get("method") in ("naive", "joint") else "-"
        ax.plot(xs, ys, style, marker="o", linewidth=2, label=label)
    # task counts and layer counts are integers; fractional ticks are nonsense
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Tasks trained so far")
    ax.set_ylabel(f"Accuracy on Task {task}")
    ax.set_title(f"What happens to Task {task} as new tasks arrive")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_drift_heatmap(runs: Dict[str, dict], out_path: str, prefer: str = "naive") -> None:
    chosen = None
    for _, res in _sorted_by_method(runs):
        if res.get("method") == prefer and "drift" in res:
            chosen = res
            break
    if chosen is None:
        for _, res in _sorted_by_method(runs):
            if "drift" in res:
                chosen = res
                break
    if chosen is None or not chosen["drift"].get("per_checkpoint"):
        return

    drift = chosen["drift"]
    layers = drift["layer_order"]
    per_cp = drift["per_checkpoint"]
    M = np.array([[cp[l]["rel_l2"] for l in layers] for cp in per_cp])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(M.T, aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xticks(range(M.shape[0]), [f"after T{i + 1}" for i in range(M.shape[0])])
    ax.set_xlabel("Checkpoint")
    ax.set_title(f"Relative weight drift from the Task-0 solution ({chosen.get('method')})")
    fig.colorbar(im, ax=ax, label=r"$\|\theta_t-\theta_0\| / \|\theta_0\|$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_ablation(runs: Dict[str, dict], out_path: str, sweep_key: str = "freeze_topk") -> None:
    """Accuracy vs layers frozen, one line per selection arm.

    This is the figure that decides whether the project has a finding. If the
    `top` line sits above `random`, `bottom` and `matched`, the drift ranking
    carries information. If the lines overlap, freezing alone explains the
    effect and the diagnosis was decorative -- which is a real result too, and
    the honest thing to plot.
    """
    from collections import defaultdict

    MODE_STYLE = {
        "top": ("tab:blue", "-", "o", "top-drift (ours)"),
        "random": ("tab:gray", "--", "^", "random layers (control)"),
        "bottom": ("tab:orange", "--", "v", "lowest-drift (control)"),
        "paramrandom": ("tab:green", "--", "s", "same #weights, random (control)"),
    }

    bucket = defaultdict(lambda: defaultdict(list))   # mode -> k -> [(acc, forg)]
    for _name, res in runs.items():
        if res.get("method") != "drift_freeze":
            continue
        info = res.get("method_info", {})
        mode = info.get("select_mode", res.get("_config", {}).get("freeze_select", "top"))
        k = _k_of(res)
        bucket[mode][k].append((res["average_accuracy"], res["average_forgetting"]))

    if not bucket:
        return

    # k=0 freezes nothing, so it is the same run for every arm; share it so
    # each line starts from the identical no-op point
    zero = None
    for mode in bucket:
        if 0 in bucket[mode]:
            zero = bucket[mode][0]
            break
    if zero is not None:
        for mode in bucket:
            bucket[mode].setdefault(0, zero)

    n_seeds = max(len(v) for m in bucket.values() for v in m.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in ["top", "random", "bottom", "paramrandom"]:
        if mode not in bucket:
            continue
        color, ls, marker, label = MODE_STYLE[mode]
        ks = sorted(bucket[mode])
        for ax, idx, ylab in ((axes[0], 0, "Average accuracy"),
                              (axes[1], 1, "Average forgetting")):
            mu = [float(np.mean([v[idx] for v in bucket[mode][k]])) for k in ks]
            sd = [float(np.std([v[idx] for v in bucket[mode][k]])) for k in ks]
            lw = 2.4 if mode == "top" else 1.4
            if n_seeds > 1:
                ax.errorbar(ks, mu, yerr=sd, fmt=marker + ls, color=color,
                            capsize=3, linewidth=lw, label=label)
            else:
                ax.plot(ks, mu, marker + ls, color=color, linewidth=lw, label=label)
            ax.set_ylabel(ylab)

    for ax in axes:
        ax.set_xlabel("Layers frozen (k)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Higher is better")
    axes[1].set_title("Lower is better")
    fig.suptitle("Does the drift ranking matter, or is freezing alone enough?"
                 + (f"  (mean of {n_seeds} seeds)" if n_seeds > 1 else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def results_table(runs: Dict[str, dict]) -> str:
    """Markdown results table, ready to paste into the 4-page report."""
    lines = [
        "| Method | Seed | Avg accuracy | Avg forgetting | BWT | Wall (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, res in _sorted_by_method(runs):
        lines.append(
            f"| {_label_for(name, res)} "
            f"| {res.get('seed', '?')} "
            f"| {res['average_accuracy']:.4f} "
            f"| {res['average_forgetting']:.4f} "
            f"| {res['backward_transfer']:+.4f} "
            f"| {res.get('wall_seconds', float('nan')):.0f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--dataset", default=None, help="plot only this dataset")
    p.add_argument("--scenario", default=None, choices=["task", "class"],
                   help="plot only this scenario")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    if not runs:
        print(f"no completed runs found in {a.runs}/")
        return

    if a.dataset:
        runs = {n: r for n, r in runs.items() if r.get("dataset") == a.dataset}
    if a.scenario:
        runs = {n: r for n, r in runs.items() if r.get("scenario") == a.scenario}

    # Overlaying runs from different datasets on one axis produces a chart
    # that silently compares unlike things. If several settings are present
    # and none was requested, plot the largest one and say so.
    settings = {}
    for n, r in runs.items():
        settings.setdefault((r.get("dataset"), r.get("scenario")), []).append(n)
    if len(settings) > 1:
        keep = max(settings.items(), key=lambda kv: len(kv[1]))
        print(f"NOTE: {len(settings)} settings found in {a.runs}/. "
              f"Plotting {keep[0][0]} / {keep[0][1]}-incremental "
              f"({len(keep[1])} runs). Use --dataset/--scenario to choose another.")
        runs = {n: runs[n] for n in keep[1]}
    if not runs:
        print("no runs left after filtering")
        return

    fig_forgetting_curve(runs, os.path.join(a.out, "fig1_forgetting_curve.png"), a.task)
    fig_drift_heatmap(runs, os.path.join(a.out, "fig2_drift_heatmap.png"))
    fig_ablation(runs, os.path.join(a.out, "fig3_ablation.png"))

    table = results_table(runs)
    table_path = os.path.join(a.out, "results_table.md")
    with open(table_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    os.replace(table_path + ".tmp", table_path)
    print(table)
    print(f"\nfigures written to {a.out}/")


if __name__ == "__main__":
    main()
