"""Forgetting-risk scoring and the stability-plasticity frontier.

Two analyses that turn claims into measurements.

**Layer-wise forgetting risk.** Drift tells you a layer *moved*. It does not
tell you that the movement caused forgetting -- a layer could churn harmlessly.
To measure the causal contribution, freeze exactly one layer, run the whole
sequence, and see how much forgetting drops relative to naive:

    risk(L) = forgetting(naive) - forgetting(freeze L alone)

A large positive risk means holding L still prevented real damage, so L was
genuinely responsible. This is an interventional measure, not a correlational
one, which is the difference between "these numbers move together" and "this
layer is the problem".

Then, and only then, is it meaningful to ask whether drift *predicts* risk.
That is a rank correlation between the two per-layer quantities, and it is the
claim the project actually rests on: if drift ranks layers the same way
intervention does, the cheap measurement is a valid proxy for the expensive one.

**Stability-plasticity frontier.** Every continual-learning method trades
retention of old tasks against the ability to learn new ones. Plotting the two
axes against each other shows the trade directly, where a single accuracy
number hides it. Both quantities are already in `results.json`:

    plasticity = mean diagonal accuracy   (how well each task was learned
                                           while it was the current task)
    stability  = mean final accuracy on every task except the last
                                          (how much of that survived)

Usage:
    python -m fmn.risk --runs runs --out figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .figures import PRETTY, load_runs  # noqa: E402


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Avoids a scipy dependency."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied ranks
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation. Returns nan for fewer than 3 points."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def stability_plasticity(res: dict) -> Tuple[float, float]:
    """(plasticity, stability) for one run."""
    diag = res.get("diagonal_accuracy") or []
    final = res.get("final_per_task_accuracy") or []
    plasticity = float(np.mean(diag)) if diag else float("nan")
    # exclude the last task: it has had no chance to be forgotten, so including
    # it inflates every method's apparent stability
    stability = float(np.mean(final[:-1])) if len(final) > 1 else float("nan")
    return plasticity, stability


def collect_risk(runs: Dict[str, dict]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Per-layer risk and drift, plus the naive forgetting reference."""
    naive = [r for r in runs.values() if r.get("method") == "naive"]
    if not naive:
        return {}, {}, float("nan")
    base_forget = float(np.mean([r["average_forgetting"] for r in naive]))

    risk: Dict[str, List[float]] = {}
    for r in runs.values():
        if r.get("method") != "drift_freeze":
            continue
        info = r.get("method_info", {})
        layers = info.get("frozen_layers", [])
        # only single-layer runs isolate one layer's contribution
        if info.get("select_mode") != "explicit" or len(layers) != 1:
            continue
        risk.setdefault(layers[0], []).append(base_forget - r["average_forgetting"])

    drift: Dict[str, float] = {}
    for r in runs.values():
        prof = r.get("method_info", {}).get("drift_profile") or {}
        if prof:
            drift = {k: float(v) for k, v in prof.items()}
            break
    if not drift:
        for r in runs.values():
            d = r.get("drift", {}).get("mean_rel_l2")
            if d:
                drift = {k: float(v) for k, v in d.items()}
                break

    return ({k: float(np.mean(v)) for k, v in risk.items()}, drift, base_forget)


def fig_risk_vs_drift(runs: Dict[str, dict], out_path: str) -> str:
    risk, drift, base = collect_risk(runs)
    layers = [l for l in risk if l in drift]
    if len(layers) < 2:
        return ("Forgetting-risk scan not run (need single-layer freeze runs). "
                "Use: python run_all.py --risk-scan --skip-baselines\n")

    xs = [drift[l] for l in layers]
    ys = [risk[l] for l in layers]
    rho = spearman(xs, ys)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.axhline(0, color="0.7", linewidth=1)
    ax.scatter(xs, ys, s=70, color="tab:blue", zorder=3)
    for l, x, y in zip(layers, xs, ys):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel(r"Mean relative weight drift  $\|\theta_t-\theta_0\|/\|\theta_0\|$")
    ax.set_ylabel("Forgetting risk\n(naive forgetting  -  forgetting when frozen alone)")
    title = "Does drift predict which layers actually cause forgetting?"
    if not np.isnan(rho):
        title += f"\nSpearman rho = {rho:+.2f}"
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    lines = [
        "| Layer | Mean drift | Forgetting risk |",
        "| --- | --- | --- |",
    ]
    for l in sorted(layers, key=lambda z: -risk[z]):
        lines.append(f"| {l} | {drift[l]:.4f} | {risk[l]:+.4f} |")
    lines.append("")
    lines.append(f"Naive average forgetting (reference): {base:.4f}")
    if not np.isnan(rho):
        lines.append(f"Spearman correlation between drift and measured risk: {rho:+.3f}")
        if rho > 0.5:
            lines.append("Drift ranks layers broadly the same way intervention does, "
                         "so it is a usable cheap proxy on this setup.")
        elif rho < -0.5:
            lines.append("Drift ranks layers in the OPPOSITE order to intervention. "
                         "Freezing the highest-drift layers is the wrong policy here.")
        else:
            lines.append("Drift and measured risk are weakly related, so drift alone "
                         "does not identify the layers responsible for forgetting.")
    return "\n".join(lines) + "\n"


def fig_frontier(runs: Dict[str, dict], out_path: str,
                 include_risk_scan: bool = False) -> None:
    """Stability against plasticity, one point per run.

    Risk-scan runs are excluded by default: they are diagnostic probes that
    freeze a single layer to measure its contribution, not candidate methods,
    and plotting them triples the point count without adding a comparison
    anyone would act on.
    """
    pts = []
    for name, res in runs.items():
        info = res.get("method_info", {})
        if info.get("select_mode") == "explicit" and not include_risk_scan:
            continue
        p, s = stability_plasticity(res)
        if np.isnan(p) or np.isnan(s):
            continue
        pts.append((res.get("method", "?"), info, p, s, name))
    if not pts:
        return

    COLOR = {"naive": "tab:red", "joint": "tab:brown", "ewc": "tab:green",
             "lwf": "tab:orange", "replay": "tab:purple", "drift_freeze": "tab:blue"}

    fig, ax = plt.subplots(figsize=(6.8, 5))
    seen = set()
    for method, info, p, s, _name in pts:
        c = COLOR.get(method, "0.5")
        label = PRETTY.get(method, method)
        marker = "o"
        if method == "drift_freeze":
            mode = info.get("select_mode", "top")
            marker = {"top": "o", "random": "^", "bottom": "v",
                      "paramrandom": "s", "explicit": "P"}.get(mode, "o")
            label += f" [{mode}]"
        ax.scatter(p, s, c=c, marker=marker, s=60,
                   label=label if label not in seen else None, zorder=3)
        seen.add(label)

    ax.set_xlabel("Plasticity  (mean accuracy on each task while it was current)")
    ax.set_ylabel("Stability  (mean final accuracy on earlier tasks)")
    ax.set_title("Stability-plasticity frontier\nup and to the right is better")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    p.add_argument("--dataset", default=None)
    p.add_argument("--scenario", default=None, choices=["task", "class"])
    p.add_argument("--include-risk-scan", action="store_true",
                   help="also plot single-layer diagnostic runs on the frontier")
    a = p.parse_args(argv)

    runs = load_runs(a.runs)
    if a.dataset:
        runs = {n: r for n, r in runs.items() if r.get("dataset") == a.dataset}
    if a.scenario:
        runs = {n: r for n, r in runs.items() if r.get("scenario") == a.scenario}
    if not runs:
        print(f"no completed runs found in {a.runs}/")
        return

    settings = {}
    for n, r in runs.items():
        settings.setdefault((r.get("dataset"), r.get("scenario")), []).append(n)
    if len(settings) > 1:
        keep = max(settings.items(), key=lambda kv: len(kv[1]))
        print(f"NOTE: {len(settings)} settings present; using {keep[0][0]} / "
              f"{keep[0][1]}-incremental. Filter with --dataset/--scenario.")
        runs = {n: runs[n] for n in keep[1]}

    os.makedirs(a.out, exist_ok=True)
    table = fig_risk_vs_drift(runs, os.path.join(a.out, "fig4_risk_vs_drift.png"))
    fig_frontier(runs, os.path.join(a.out, "fig5_stability_plasticity.png"),
                 include_risk_scan=a.include_risk_scan)

    dest = os.path.join(a.out, "risk_table.md")
    with open(dest + ".tmp", "w", encoding="utf-8") as f:
        f.write(table)
    os.replace(dest + ".tmp", dest)
    print(table)
    print(f"written to {dest}")


if __name__ == "__main__":
    main()
