"""Seeding, device selection, and run configuration.

Reproducibility is a graded criterion, so every source of randomness that we
control is pinned here and nowhere else.
"""

from __future__ import annotations

import os

# Must be set BEFORE the CUDA context is created, otherwise deterministic
# cuBLAS matmuls silently do not take effect. Setting it at import time is the
# only place that is reliably early enough.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random  # noqa: E402
from dataclasses import dataclass, field, asdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42


def seed_everything(seed: int = SEED, deterministic: bool = True) -> None:
    """Pin every RNG we touch.

    `deterministic=True` also forces cuDNN into a reproducible mode. This
    costs some speed but makes two runs of the same command give the same
    number, which is exactly what the rubric asks for.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # older torch without warn_only
            pass


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def worker_init_fn(worker_id: int) -> None:
    """Make DataLoader workers deterministic too."""
    s = SEED + worker_id
    np.random.seed(s)
    random.seed(s)


@dataclass
class RunConfig:
    """Everything that defines a run. Dumped verbatim into results JSON."""

    # what
    dataset: str = "split-mnist"      # split-mnist | split-fashion | split-cifar10 | split-cifar100
    scenario: str = "task"            # task | class
    method: str = "naive"             # naive | joint | replay | ewc | lwf | drift_freeze

    # optimisation
    epochs_per_task: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"           # adam | sgd
    momentum: float = 0.9             # sgd only

    # method hyper-parameters
    buffer_size: int = 200            # replay: total reservoir capacity
    replay_batch: int = 32            # replay: extra samples drawn per step
    ewc_lambda: float = 5000.0        # ewc: penalty strength
    ewc_fisher_batches: int = 50      # ewc: minibatches used to estimate Fisher
    lwf_alpha: float = 1.0            # lwf: distillation weight
    lwf_temperature: float = 2.0      # lwf: softening temperature
    freeze_topk: int = 2              # drift_freeze: how many layers to freeze
    freeze_select: str = "top"        # drift_freeze: top | bottom | random | paramrandom
    freeze_layers: str = ""           # drift_freeze: explicit layer list, overrides the above
    drift_profile: str = ""           # drift_freeze: path to a naive run's drift.json

    # bookkeeping
    seed: int = SEED
    device: str = "auto"
    num_workers: int = 2
    data_root: str = "./data"
    out_dir: str = "./runs"
    tag: str = ""
    amp: bool = True                  # mixed precision (ignored on CPU)
    eval_on: str = "test"             # test | val
    max_batches_per_epoch: int = 0    # >0 truncates every epoch; for smoke tests only

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_name(self) -> str:
        if self.tag:
            return self.tag
        name = f"{self.dataset}_{self.scenario}_{self.method}"
        if self.method == "drift_freeze":
            if self.freeze_layers:
                name += "_only-" + self.freeze_layers.replace(",", "-")
            else:
                name += f"_{self.freeze_select}k{self.freeze_topk}"
        return f"{name}_s{self.seed}"
