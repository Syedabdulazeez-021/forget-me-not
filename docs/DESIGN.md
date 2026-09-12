# Forget-Me-Not: Locating and Freezing the Layers That Forget

**Track 5 — Continual Learning**

A neural network trained on a sequence of tasks forgets the earlier ones.
Everyone knows *that* it happens. This project asks **where in the network it
happens**, measures it, and then uses the measurement to build a method.

Two stages:

1. **Diagnose.** Train naively through the task sequence, snapshotting weights
   after every task, and measure how far each layer group moved away from the
   state in which Task 0 was solved.
2. **Act.** Freeze the top-*k* highest-drift layers from Task 1 onward and see
   whether forgetting drops. Sweeping *k* is the ablation.

The method costs no extra memory, no stored exemplars, and no second forward
pass — which is the interesting part if it works, and an honest negative
result if it doesn't.

---

## Quick start

```bash
pip install -r requirements.txt        # install torch for your CUDA first
python scripts/smoke_test.py           # ~1 min, validates every code path
python run_all.py                      # full reproduction
```

`run_all.py` is the one-command entry point the rules require. It runs both
mandatory baselines, three published reference methods, the four-point
ablation, and writes the figures and both results tables. It is plain Python,
so it works identically on Windows, macOS and Linux — no bash needed.

```bash
python run_all.py --dataset split-cifar10 --epochs 5 --seeds 42 43 44
python run_all.py --quick              # pipeline check, ~1 min
python run_all.py --dry-run            # print the commands without running
```

`run_all.sh` and `scripts/smoke_test.sh` are thin wrappers around the two
Python scripts, for people already in a shell. On Windows use the `.py` files
directly.

---

## What gets produced

| Path | Contents |
| --- | --- |
| `runs/<tag>/config.json` | exact configuration, seed included |
| `runs/<tag>/acc_matrix.csv` | `R[i][j]` = accuracy on task *j* after task *i* |
| `runs/<tag>/results.json` | average accuracy, average forgetting, BWT, wall time |
| `runs/<tag>/drift.json` | per-layer drift, per checkpoint |
| `runs/<tag>/log.txt` | full trace |
| `figures/fig1_forgetting_curve.png` | Task-0 accuracy as tasks arrive — **the demo slide** |
| `figures/fig2_drift_heatmap.png` | per-layer drift — the diagnosis |
| `figures/fig3_ablation.png` | accuracy and forgetting vs k, **all four arms** — the falsification figure |
| `figures/fig4_risk_vs_drift.png` | measured per-layer forgetting risk against drift, with Spearman rho |
| `figures/fig5_stability_plasticity.png` | stability-plasticity frontier, one point per method |
| `figures/risk_table.md` | per-layer drift and risk, and what the correlation means |
| `figures/results_table.md` | per-run table, paste straight into the report |
| `figures/aggregate.md` | mean ± std across seeds, plus an explicit "did it beat the baseline?" table |

---

## Methods

| `--method` | What it is | Role |
| --- | --- | --- |
| `naive` | sequential fine-tuning | **mandatory lower bound** |
| `joint` | cumulative training on all tasks seen so far | **mandatory upper bound** |
| `ewc` | Elastic Weight Consolidation, diagonal empirical Fisher | reference |
| `lwf` | Learning without Forgetting, KD from the previous model | reference |
| `replay` | experience replay, reservoir buffer | reference |
| `drift_freeze` | **ours** — freeze the highest-drift layers | contribution |

`drift_freeze` is two-stage by construction: it consumes the `drift.json`
written by a `naive` run. `--freeze-topk 0` is a deliberate no-op that must
reproduce the naive numbers exactly; the smoke test checks this, and it is the
*k*=0 point of the ablation curve.

## The control arms — the actual contribution

Freezing the highest-drift layers and observing less forgetting proves
nothing on its own. Freezing *anything* reduces plasticity, and the
highest-drift layers are usually also the largest. `--freeze-select` runs the
comparison that separates those explanations:

| `--freeze-select` | Freezes | Answers |
| --- | --- | --- |
| `top` | the k highest-drift layers | the method |
| `bottom` | the k lowest-drift layers | does the *sign* of the drift signal mean anything? |
| `random` | k layers at random | does the *ranking* beat freezing per se? |
| `paramrandom` | the same **number of weights** as `top`, scattered at random across the network | does *which* layers matter, or just *how many weights*? |

`paramrandom` is the strict one. Since layer sizes are wildly unequal, a
layer-level parameter match is impossible — no disjoint subset of layers comes
near the top-k budget, and the closest match is always top-k plus a tiny
layer, which is the method again. So the control operates at weight
granularity instead: a fixed random mask over individual weights, matched
exactly on count, with those gradients zeroed before each optimiser step.

If `top` separates from all three controls, the drift measurement carries
information and you have a finding. If the lines overlap, freezing alone
explains the effect and the diagnosis was decorative — also a real result, and
the honest one to report. `fig3_ablation.png` is that comparison.

Skip the controls with `--no-controls` if you are short on time, but the
result is then not falsifiable.

## Scenarios — read this before your first real run

- `--scenario class` — class-incremental. Logits span every class seen so far.
  Naive fine-tuning collapses. **This is the setting to use.**
- `--scenario task` (default) — task-incremental. Logits are masked to the
  task's own classes.

On Split-MNIST with `--scenario task`, naive fine-tuning reaches ~0.988 average
accuracy with ~0.012 forgetting, against a joint ceiling of ~0.997. The whole
floor-to-ceiling gap is under one percent, so there is essentially nothing for
any method to repair, and freezing layers only removes capacity the model
needed. Measured, not hypothetical.

If your floor does not fail, you have no experiment. Check the naive forgetting
number first; if it is near zero, switch to `--scenario class` or a harder
dataset before running anything else.

## Choosing a dataset

If the dataset is yours to pick, two constraints decide it.

**The naive floor must actually fail.** Measured on Split-MNIST task-IL: naive
0.988, joint 0.997. No headroom, no experiment. Check the naive forgetting
number before running anything else.

**The backbone needs enough layers to rank.** SmallCNN (MNIST, FashionMNIST)
exposes three freezable groups; a "which layers forget" study across three
layers is thin, and k exhausts the network almost immediately. The CIFAR
datasets use ReducedResNet18 — stem plus four stages, five candidates — so the
ranking has something to say.

Both point to **`--dataset split-cifar10 --scenario class`** as the default,
with `split-cifar100` (10 tasks, same backbone) as the generalisation check if
time allows.

## Forgetting-risk scan — does drift actually predict forgetting?

Drift tells you a layer *moved*. It does not tell you the movement caused
forgetting; a layer could churn harmlessly. The risk scan measures the causal
contribution directly by freezing exactly one layer and running the whole
sequence:

    risk(L) = forgetting(naive) - forgetting(freeze L alone)

A large positive risk means holding L still prevented real damage. This is an
interventional measure, not a correlational one — the difference between
"these numbers move together" and "this layer is the problem".

```bash
python run_all.py --risk-scan --skip-baselines --dataset split-cifar10 --scenario class
```

One run per candidate layer (five on the CIFAR backbone). `fig4_risk_vs_drift.png`
then plots measured risk against drift and reports the Spearman rank
correlation. High positive rho means the cheap measurement is a valid proxy for
the expensive one, which is the claim the whole method rests on. Near zero, or
negative, means it is not — and `risk_table.md` says so in plain language
rather than leaving it for a reader to notice.

## Stability-plasticity frontier

Every continual-learning method trades retention against the ability to learn
new tasks; a single accuracy number hides that trade.
`fig5_stability_plasticity.png` plots both axes, one point per method:

    plasticity = mean diagonal accuracy (each task while it was current)
    stability  = mean final accuracy on every task except the last

It needs no extra runs — both quantities are already in every `results.json`.
Risk-scan probes are excluded by default (`--include-risk-scan` to show them);
they are diagnostics, not candidate methods.

## Budgeting runs

A full `run_all.py` is 18 runs per seed: 5 baselines, plus k=0 once, plus
3 values of k times 4 selection arms. Three seeds is 54 runs — too many for an
8-hour window on CIFAR. Split it:

```bash
# phase 1 -- baselines + the top-k sweep only (9 runs)
python run_all.py --dataset split-cifar10 --scenario class --epochs 5 --no-controls

# read fig3, pick the best k, then run the controls at that k only (3 runs)
python run_all.py --dataset split-cifar10 --scenario class --epochs 5 \
    --skip-baselines --freeze-ks 2 --selects random bottom paramrandom

# extra seeds for error bars on the comparison that matters
python run_all.py --dataset split-cifar10 --scenario class --epochs 5 \
    --seeds 43 44 --freeze-ks 2
```

`--skip-baselines` reuses the baselines and drift profile already in the
out-dir instead of retraining them.

## Keep run directories clean

`--out-dir` defaults to `runs/`, and every run in there is picked up by the
aggregator. Short throwaway runs (dataset downloads, pipeline checks) must go
somewhere else:

```bash
python -m fmn.train --dataset split-cifar10 --max-batches-per-epoch 2 --out-dir throwaway
```

Mixing settings in one directory used to make the aggregator compare a method
on one dataset against a baseline from another. It now splits the report by
(dataset, scenario) and says so, and `--dataset` / `--scenario` filter both the
tables and the figures. But the cleanest fix is not to mix them.

## Metrics

Reported exactly as the track brief specifies, from the accuracy matrix `R`:

- **Average accuracy** — mean of the final row.
- **Average forgetting** — mean over tasks *j* < *T*−1 of (best accuracy on *j*
  before the end) − (accuracy on *j* at the end). Chaudhry et al., 2018.
- **Backward transfer (BWT)** — mean of `R[T-1][j] − R[j][j]`. Lopez-Paz &
  Ranzato, 2017.

The metric code is pure NumPy and unit-tested (`tests/test_metrics.py`, 8 tests)
so the scoring cannot drift when the training code changes. The selection
policy is likewise torch-free and unit-tested (`tests/test_selection.py`,
13 tests) — including that the control arms actually differ from the method,
which is the assumption the whole claim rests on. The risk and frontier
statistics are tested too (`tests/test_risk.py`, 14 tests), since the Spearman
number in fig4 is the sub-problem's headline result. 35 tests in total.

## Reproducibility

- `seed=42` everywhere by default; `--seed` to change it.
- Python, NumPy, torch, and CUDA RNGs are all pinned in `fmn/config.seed_everything`.
- cuDNN is put in deterministic mode and `torch.use_deterministic_algorithms`
  is enabled.
- DataLoader workers are seeded via `worker_init_fn` and an explicit generator.
- Class order and the train/val carve-out use their own fixed seeds,
  independent of the run seed, so every method sees byte-identical splits.
- **The test split is used for the final number only.** Use `--eval-on val`
  for any tuning.

## Layout

```
fmn/
  config.py    seeding, device, RunConfig
  data.py      task-sequence construction, fixed splits
  models.py    SmallCNN (28x28 grey), ReducedResNet18 (32x32 RGB)
  methods.py   the six strategies + reservoir buffer
  drift.py     per-layer drift measurement
  selection.py layer/weight selection policy (torch-free, unit-tested)
  metrics.py   avg accuracy / forgetting / BWT  (pure NumPy, tested)
  train.py     the training loop
  figures.py   figures 1-3 + per-run results table
  risk.py      forgetting-risk scoring, frontier, Spearman correlation
  aggregate.py mean ± std across seeds + baseline comparison table
run_all.py         <- one-command entry point (cross-platform)
run_all.sh         <- shell wrapper
scripts/smoke_test.py   <- run this first
scripts/smoke_test.sh   <- shell wrapper
tests/test_metrics.py     8 tests
tests/test_selection.py   13 tests
tests/test_risk.py        14 tests
```

Adding a dataset = one entry in `fmn/data._SPECS`. Adding a method = one
subclass in `fmn/methods.py` plus a line in `REGISTRY`.

## Correctness details worth knowing

**Per-sample logit masking.** A batch is not always single-task: experience
replay splices in samples from earlier tasks, and joint training draws from
every task at once. Masking the whole batch with the *current* task's classes
would force replayed samples' own correct labels onto a near-infinite negative
logit, making their loss enormous and their gradients meaningless — silently
turning the strongest baseline into a broken one. Because every class belongs
to exactly one task, a sample's label recovers its task, so `LogitMasker` masks
each row with the classes of the task it actually came from.

**EWC Fisher under the same mask.** The Fisher is estimated with the identical
masking used during training. Estimating it on unmasked logits would measure
sensitivity of a distribution the model never optimised.

**Frozen BatchNorm.** Freezing a BatchNorm's weight and bias is not enough —
running mean and variance are buffers, not parameters, and keep absorbing
new-task statistics regardless of `requires_grad`. `DriftFreeze` puts fully
frozen BatchNorm modules into `eval()` mode, resolved by parameter ownership
rather than attribute name (name matching silently missed the stem BatchNorm
in `ReducedResNet18`). `model.train()` undoes this at every epoch boundary, so
the `on_train_mode()` hook re-asserts it.

**cuBLAS determinism.** `CUBLAS_WORKSPACE_CONFIG` is set at import time in
`fmn/config.py`, before any CUDA context can exist. Setting it later has no
effect.

## Notes and known limits

- `--no-amp` if EWC's penalty destabilises under mixed precision; a large
  `--ewc-lambda` with fp16 can overflow the loss scaler.
- `drift_freeze` puts frozen BatchNorm modules in `eval()` mode. Without this,
  running statistics keep absorbing new-task data and leak into old-task
  predictions even though the weights are frozen.
- The classifier head is never a freeze candidate — freezing it would make new
  classes unlearnable. Enforced in `drift.rank_layers_by_drift`.
- Drift is measured on parameters only; BatchNorm buffers are excluded.

## Citations

- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*, PNAS — EWC.
- Li & Hoiem (2016), *Learning without Forgetting*, ECCV — LwF.
- Lopez-Paz & Ranzato (2017), *Gradient Episodic Memory for Continual Learning*, NeurIPS — BWT metric, reduced-ResNet18 backbone.
- Chaudhry et al. (2018), *Riemannian Walk for Incremental Learning*, ECCV — forgetting metric.
- Vitter (1985), *Random sampling with a reservoir*, ACM TOMS — reservoir buffer.
- Datasets: MNIST (LeCun et al.), FashionMNIST (Xiao et al., 2017), CIFAR-10/100 (Krizhevsky, 2009), via `torchvision.datasets`.
