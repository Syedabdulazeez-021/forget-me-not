# Forget-Me-Not: Locating and Freezing the Layers That Forget

**Deep Learning Hackathon — Track 5: Continual Learning**

## Team

- Syed Abdul Azeez (AM.SC.U4AIE24049) — team lead
- Saripalli Sri Nandan (AM.SC.U4AIE24043)
- Dukka Chanakya Tej (AM.SC.U4AIE24070)

## What this project asks

A neural network trained on a sequence of tasks forgets the earlier ones —
that much is well known. This project asks *where in the network* that
forgetting happens: does it concentrate in a few layers, or spread evenly
across all of them? And if it does concentrate, is that measurable signal
actually actionable — can freezing the layers that moved the most reduce
forgetting, or does freezing *any* layers work just as well? The project
builds an interventional risk measure to test that question directly, rather
than assuming a correlation implies a cause.

## Key results

| Method | Avg accuracy | Avg forgetting | BWT |
| --- | --- | --- | --- |
| Naive fine-tuning (lower bound) | 0.1805 | 0.9557 | -0.9557 |
| Cumulative joint (upper bound) | 0.8161 | 0.0226 | -0.0149 |
| Drift-freeze, select=top, k=1 | 0.1726 | 0.9484 | -0.9484 |
| Drift-freeze, select=top, k=2 | 0.1695 | 0.9427 | -0.9427 |
| Drift-freeze, select=random, k=2 | 0.1782 | 0.9456 | -0.9456 |
| Drift-freeze, select=bottom, k=2 | 0.1798 | 0.9470 | -0.9470 |

Full table with all four selection arms, all k, and both reference methods
(EWC, LwF, replay) is in [figures/aggregate.md](figures/aggregate.md).

Per-layer drift and measured forgetting risk, from freezing exactly one layer
at a time:

| Layer | Mean drift | Forgetting risk |
| --- | --- | --- |
| layer2 | 0.3732 | +0.0179 |
| layer4 | 0.6005 | +0.0074 |
| layer3 | 0.5135 | +0.0045 |
| layer1 | 0.2990 | +0.0031 |
| stem | 0.1067 | +0.0009 |

Full table in [figures/risk_table.md](figures/risk_table.md).

**Findings, stated directly:** the drift-based ranking gave no real advantage
over freezing layers at random — the `top`, `random`, and `bottom` selection
arms land within a few thousandths of each other across every k. And the two
signals disagree on which layer matters most: drift magnitude ranks `layer4`
highest, while the interventional risk measure — actually freezing each layer
alone and measuring the change in forgetting — identifies `layer2` as the most
damaging to freeze. Drift tells you a layer moved; it does not reliably tell
you that movement caused the forgetting.

## Quickstart

```bash
pip install -r requirements.txt        # install torch for your CUDA first
python scripts/smoke_test.py           # ~1 min, validates every code path
python run_all.py --dataset split-cifar10 --scenario class --epochs 5
```

`run_all.py` is the one-command entry point. It runs both mandatory
baselines, three reference methods, the four-arm ablation, and writes the
figures and results tables. It is plain Python, so it runs identically on
Windows, macOS, and Linux.

## What's in the repo

```
fmn/            core library: data, models, methods, drift, selection, metrics, training, figures, risk, aggregation
tests/          35 unit tests (metrics, selection, risk/frontier statistics)
scripts/        smoke_test.py / .sh — fast pipeline check
run_all.py      one-command reproduction entry point (cross-platform)
run_all.sh      shell wrapper around run_all.py
figures/        generated PNGs and markdown result tables (checked in)
runs/           per-run config.json, results.json, acc_matrix.csv, drift.json, log.txt (checked in)
docs/DESIGN.md  full design documentation — methods, control arms, scenarios, correctness notes
```

## Experimental design

The contribution is `drift_freeze`: freeze the highest-drift layers from
Task 1 onward. On its own, freezing highest-drift layers proving less
forgetting doesn't prove anything — freezing *anything* reduces plasticity,
and the highest-drift layers are often the largest. Four selection arms
isolate what's actually doing the work:

| `--freeze-select` | Freezes | Controls for |
| --- | --- | --- |
| `top` | the k highest-drift layers | the method itself |
| `bottom` | the k lowest-drift layers | whether the *sign* of the drift signal means anything |
| `random` | k layers at random | whether the *ranking* beats freezing per se |
| `paramrandom` | the same number of weights as `top`, scattered at random | whether it's *which* layers, or just *how many weights* |

`--freeze-topk 0` is a deliberate no-op arm: it must reproduce the naive
numbers exactly, and it is the k=0 point on the ablation curve — a sanity
check that the freezing mechanism itself introduces no side effects.

The forgetting-risk scan is the interventional measure: freeze exactly one
layer at a time, run the full task sequence, and compute

    risk(L) = forgetting(naive) - forgetting(freeze L alone)

against the correlational drift signal. Comparing the two rankings is what
lets the project report an honest answer rather than assuming drift implies
causation.

## Reproducibility

- `seed=42` everywhere by default (`--seed` to change it).
- Python, NumPy, torch, and CUDA RNGs are all pinned (`fmn/config.seed_everything`).
- Class order and train/val/test splits are fixed independently of the run
  seed, so every method sees byte-identical data.
- 35 unit tests cover the metrics, the selection policy, and the risk/frontier
  statistics.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design documentation:
scenario choice, dataset choice, run budgeting, and correctness details
(per-sample logit masking, frozen BatchNorm handling, cuBLAS determinism).

## References

- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*, PNAS — EWC.
- Li & Hoiem (2016), *Learning without Forgetting*, ECCV — LwF.
- Lopez-Paz & Ranzato (2017), *Gradient Episodic Memory for Continual Learning*, NeurIPS — BWT metric, reduced-ResNet18 backbone.
- Chaudhry et al. (2018), *Riemannian Walk for Incremental Learning*, ECCV — forgetting metric.
- Vitter (1985), *Random sampling with a reservoir*, ACM TOMS — reservoir buffer.
- Datasets: MNIST (LeCun et al.), FashionMNIST (Xiao et al., 2017), CIFAR-10/100 (Krizhevsky, 2009), via `torchvision.datasets`.
