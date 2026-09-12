#!/usr/bin/env python
"""Run this FIRST, on the prep day, before the clock starts.

Exercises every code path -- all six methods, both scenarios, the drift-profile
handoff, the figure pipeline and the aggregator -- with tiny epochs, so it
finishes in about a minute on CPU. The accuracy numbers it produces are
meaningless. The point is that nothing crashes at 2pm on hackathon day.

    python scripts/smoke_test.py

Works on Windows, macOS and Linux.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "smoke_runs")
FIGS = os.path.join(ROOT, "smoke_figures")
PY = sys.executable

COMMON = [
    "--dataset", "split-mnist",
    "--epochs-per-task", "1",
    "--max-batches-per-epoch", "3",
    "--out-dir", OUT,
    "--num-workers", "0",
    "--device", "cpu",
]

failures: list[str] = []
RISK_TAGS: list[str] = []


def step(title: str) -> None:
    print(f"\n{'=' * 58}\n{title}\n{'=' * 58}", flush=True)


def run(args, label: str, quiet: bool = True) -> bool:
    r = subprocess.run(args, cwd=ROOT,
                       stdout=subprocess.PIPE if quiet else None,
                       stderr=subprocess.STDOUT if quiet else None)
    ok = r.returncode == 0
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}", flush=True)
    if not ok:
        failures.append(label)
        if quiet and r.stdout:
            tail = r.stdout.decode("utf-8", "replace").strip().splitlines()[-15:]
            print("        " + "\n        ".join(tail))
    return ok


def train(extra, tag: str) -> bool:
    return run([PY, "-m", "fmn.train"] + COMMON + extra + ["--tag", tag], tag)


def check(cond: bool, label: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}", flush=True)
    if not cond:
        failures.append(label)


def results(tag: str) -> dict:
    with open(os.path.join(OUT, tag, "results.json"), encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree(FIGS, ignore_errors=True)

    step("1. unit tests")
    run([PY, "-m", "pytest", "tests", "-q"], "pytest tests", quiet=False)

    step("2. task-incremental: every method")
    for m in ["naive", "joint", "ewc", "lwf", "replay"]:
        train(["--method", m], f"smoke_{m}")

    step("3. drift-freeze handoff")
    profile = os.path.join(OUT, "smoke_naive", "drift.json")
    check(os.path.isfile(profile), "naive wrote drift.json")
    if os.path.isfile(profile):
        train(["--method", "drift_freeze", "--freeze-topk", "0",
               "--drift-profile", profile], "smoke_df_k0")
        for mode in ["top", "bottom", "random", "paramrandom"]:
            train(["--method", "drift_freeze", "--freeze-topk", "2",
                   "--freeze-select", mode, "--drift-profile", profile],
                  f"smoke_df_{mode}_k2")

    step("4. forgetting-risk scan (single-layer freezing)")
    if os.path.isfile(profile):
        import json as _json
        with open(profile, encoding="utf-8") as f:
            layers = [l for l in _json.load(f).get("layer_order", []) if l != "head"]
        for layer in layers[:2]:      # two is enough to prove the path works
            train(["--method", "drift_freeze", "--freeze-layers", layer,
                   "--drift-profile", profile], f"smoke_risk_{layer}")
        RISK_TAGS.extend(f"smoke_risk_{l}" for l in layers[:2])

    step("5. class-incremental spot check")
    train(["--scenario", "class", "--method", "naive"], "smoke_class_naive")
    train(["--scenario", "class", "--method", "replay"], "smoke_class_replay")

    step("6. artefacts present")
    tags = ["smoke_naive", "smoke_joint", "smoke_ewc", "smoke_lwf", "smoke_replay",
            "smoke_df_k0", "smoke_df_top_k2", "smoke_df_bottom_k2",
            "smoke_df_random_k2", "smoke_df_paramrandom_k2",
            "smoke_class_naive", "smoke_class_replay"] + RISK_TAGS
    for t in tags:
        for f in ("results.json", "acc_matrix.csv", "drift.json", "config.json", "log.txt"):
            check(os.path.isfile(os.path.join(OUT, t, f)), f"{t}/{f}")

    step("7. figures and aggregation")
    run([PY, "-m", "fmn.figures", "--runs", OUT, "--out", FIGS], "figures")
    run([PY, "-m", "fmn.aggregate", "--runs", OUT,
         "--out", os.path.join(FIGS, "aggregate.md")], "aggregate")
    run([PY, "-m", "fmn.risk", "--runs", OUT, "--out", FIGS], "risk analysis")
    for f in ("fig1_forgetting_curve.png", "fig2_drift_heatmap.png",
              "fig5_stability_plasticity.png", "results_table.md",
              "aggregate.md", "risk_table.md"):
        fp = os.path.join(FIGS, f)
        # non-empty, not merely present: a crash mid-write leaves a 0-byte
        # or truncated file that a bare isfile() check happily accepts
        check(os.path.isfile(fp) and os.path.getsize(fp) > 0, f"figures/{f}")

    step("8. correctness assertions")
    try:
        naive, k0 = results("smoke_naive"), results("smoke_df_k0")
        same = abs(naive["average_accuracy"] - k0["average_accuracy"]) < 1e-6
        check(same, "drift_freeze k=0 exactly reproduces naive (no-op check)")
        if not same:
            print(f"        naive={naive['average_accuracy']:.6f} "
                  f"k0={k0['average_accuracy']:.6f}")
    except Exception as e:  # noqa: BLE001
        check(False, f"k=0 no-op check ({e})")

    # The control arms only mean something if they actually differ from the
    # method. If `top` and `bottom` froze the same layers, or paramrandom
    # froze a different number of weights than top, the comparison in the
    # report would be vacuous while still producing a tidy-looking plot.
    try:
        top = results("smoke_df_top_k2")["method_info"]
        bot = results("smoke_df_bottom_k2")["method_info"]
        rnd = results("smoke_df_random_k2")["method_info"]
        prm = results("smoke_df_paramrandom_k2")["method_info"]
        check(set(top["frozen_layers"]) != set(bot["frozen_layers"]),
              "bottom control freezes different layers than top")
        check(len(rnd["frozen_layers"]) == len(top["frozen_layers"]),
              "random control freezes the same NUMBER of layers as top")
        check(prm["freeze_granularity"] == "weights",
              "paramrandom control operates at weight granularity")
        check(prm["frozen_param_count"] == top["frozen_param_count"],
              "paramrandom control freezes the same NUMBER of weights as top")
        print(f"        top froze {top['frozen_param_count']:,} weights "
              f"in layers {top['frozen_layers']}")
        print(f"        paramrandom froze {prm['frozen_param_count']:,} "
              f"weights scattered across the network")
    except Exception as e:  # noqa: BLE001
        check(False, f"control-arm distinctness ({e})")

    # Replay mixes old-task samples into the batch. If per-sample logit
    # masking were wrong, those samples would land on a -1e9 logit for their
    # own true label and the loss would blow up. A finite, sane loss is the
    # cheap signal that the masking is right.
    try:
        rep = results("smoke_replay")
        finite = all(0.0 <= v <= 1.0 for v in rep["final_per_task_accuracy"])
        check(finite, "replay accuracies are in [0,1] (per-sample masking sane)")
        with open(os.path.join(OUT, "smoke_replay", "log.txt"), encoding="utf-8") as f:
            txt = f.read()
        check("nan" not in txt.lower(), "replay log contains no NaN loss")
    except Exception as e:  # noqa: BLE001
        check(False, f"replay sanity ({e})")

    try:
        for t in tags:
            import csv
            with open(os.path.join(OUT, t, "acc_matrix.csv"), encoding="utf-8") as f:
                rows = [r for r in csv.reader(f) if r]
            check(len(rows) == len(rows[0]), f"{t} accuracy matrix is square")
    except Exception as e:  # noqa: BLE001
        check(False, f"matrix shape ({e})")

    print("\n" + "=" * 58)
    if failures:
        print(f"SMOKE TEST FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED -- every path works. Safe to run run_all.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
