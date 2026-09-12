#!/usr/bin/env python
"""One-command reproduction. Works on Windows, macOS and Linux.

    python run_all.py
    python run_all.py --dataset split-cifar10 --epochs 5 --seeds 42 43 44
    python run_all.py --quick          # tiny run, checks the pipeline end to end

Stage 1 runs both mandatory baselines plus three published reference methods.
Stage 2 reads the naive run's drift profile and sweeps how many layers to
freeze -- that sweep is the required ablation.
Stage 3 draws the figures and writes the results tables.

This is the canonical entry point. `run_all.sh` is a thin convenience wrapper
for people already in a shell; it does the same thing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def sh(args, dry=False) -> None:
    printable = " ".join(args[2:])  # drop "python -m"
    print(f"\n$ python -m {printable}", flush=True)
    if dry:
        return
    t0 = time.time()
    r = subprocess.run(args)
    if r.returncode != 0:
        print(f"\nFAILED (exit {r.returncode}): {' '.join(args)}", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="split-mnist")
    p.add_argument("--scenario", default="task", choices=["task", "class"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--freeze-ks", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--selects", nargs="+", default=["top", "random", "bottom", "paramrandom"],
                   choices=["top", "bottom", "random", "paramrandom"],
                   help="top is the method; the rest are falsification controls")
    p.add_argument("--no-controls", action="store_true",
                   help="run only the top-k arm (faster, but the result is not provable)")
    p.add_argument("--risk-scan", action="store_true",
                   help="freeze each candidate layer ALONE to measure its "
                        "individual contribution to forgetting (one run per layer)")
    p.add_argument("--skip-baselines", action="store_true",
                   help="reuse baselines already in --out-dir and run only the "
                        "drift-freeze arms; for phase-2 runs after a sweep")
    p.add_argument("--buffer-size", type=int, default=200)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--quick", action="store_true",
                   help="1 epoch, 3 batches per epoch -- pipeline check only")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    py = sys.executable  # the interpreter running this file, venv-safe
    base = [py, "-m", "fmn.train",
            "--dataset", a.dataset, "--scenario", a.scenario,
            "--epochs-per-task", str(1 if a.quick else a.epochs),
            "--out-dir", a.out_dir, "--device", a.device,
            "--num-workers", str(a.num_workers)]
    if a.quick:
        base += ["--max-batches-per-epoch", "3"]
    if a.no_amp:
        base += ["--no-amp"]

    t_start = time.time()
    print("=" * 60)
    print(" Forget-Me-Not: Locating and Freezing the Layers That Forget")
    print(f" dataset={a.dataset} scenario={a.scenario} "
          f"epochs={1 if a.quick else a.epochs} seeds={a.seeds}")
    print("=" * 60)

    for seed in a.seeds:
        suf = f"{a.dataset}_{a.scenario}_s{seed}"
        seeded = base + ["--seed", str(seed)]

        if a.skip_baselines:
            print(f"\n### [1/3] baselines skipped (reusing {a.out_dir}/, seed {seed})")
        else:
            print(f"\n### [1/3] baselines and reference methods (seed {seed})")
            # naive is first on purpose: it writes the drift profile stage 2 needs
            sh(seeded + ["--method", "naive", "--tag", f"naive_{suf}"], a.dry_run)
            sh(seeded + ["--method", "joint", "--tag", f"joint_{suf}"], a.dry_run)
            sh(seeded + ["--method", "ewc", "--tag", f"ewc_{suf}"], a.dry_run)
            sh(seeded + ["--method", "lwf", "--tag", f"lwf_{suf}"], a.dry_run)
            sh(seeded + ["--method", "replay", "--buffer-size", str(a.buffer_size),
                         "--tag", f"replay_{suf}"], a.dry_run)

        print(f"\n### [2/3] drift-freeze ablation + controls (seed {seed})")
        profile = os.path.join(a.out_dir, f"naive_{suf}", "drift.json")
        if not a.dry_run and not os.path.isfile(profile):
            print(f"ERROR: expected drift profile at {profile}", file=sys.stderr)
            if a.skip_baselines:
                print("       --skip-baselines needs a naive run already present "
                      "in this out-dir, with matching dataset/scenario/seed.",
                      file=sys.stderr)
            sys.exit(1)

        if a.risk_scan:
            # one run per layer, each freezing that layer and nothing else.
            # The drop in forgetting versus naive is that layer's risk score.
            import json as _json
            if a.dry_run and not os.path.isfile(profile):
                candidates = ["<layers read from the naive run's drift.json>"]
            else:
                with open(profile, encoding="utf-8") as f:
                    prof = _json.load(f)
                candidates = [l for l in prof.get("layer_order", []) if l != "head"]
            print(f"  risk scan over {len(candidates)} layers: {candidates}")
            for layer in candidates:
                sh(seeded + ["--method", "drift_freeze", "--drift-profile", profile,
                             "--freeze-layers", layer,
                             "--tag", f"risk_{layer}_{suf}"], a.dry_run)
            continue

        selects = ["top"] if a.no_controls else a.selects
        for k in a.freeze_ks:
            for mode in selects:
                # k=0 freezes nothing, so every mode collapses to the same
                # run; doing it once avoids three identical rows
                if k == 0 and mode != selects[0]:
                    continue
                sh(seeded + ["--method", "drift_freeze", "--drift-profile", profile,
                             "--freeze-topk", str(k), "--freeze-select", mode,
                             "--tag", f"df_{mode}_k{k}_{suf}"], a.dry_run)

    print("\n### [3/3] figures and tables")
    sh([py, "-m", "fmn.figures", "--runs", a.out_dir, "--out", "figures"], a.dry_run)
    sh([py, "-m", "fmn.aggregate", "--runs", a.out_dir,
        "--out", "figures/aggregate.md"], a.dry_run)
    sh([py, "-m", "fmn.risk", "--runs", a.out_dir, "--out", "figures"], a.dry_run)

    mins = (time.time() - t_start) / 60
    print(f"\nDone in {mins:.1f} min.")
    print("  figures/          fig1 forgetting curve, fig2 drift heatmap,")
    print("                    fig3 ablation + controls, fig4 risk vs drift,")
    print("                    fig5 stability-plasticity frontier")
    print("  figures/aggregate.md   results table + explicit baseline comparison")
    print(f"  {a.out_dir}/            per-run configs, accuracy matrices, logs")


if __name__ == "__main__":
    main()
