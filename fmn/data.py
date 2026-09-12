"""Task-sequence construction with fixed, seed-locked splits.

Design decisions that matter for the report:

* The **official** train/test split of each dataset is respected. We never
  touch the test split except for the final number.
* A 10% validation slice is carved out of *train* with a fixed generator so
  that it is byte-identical across runs and across methods.
* Tasks are formed by partitioning the label space into contiguous chunks
  (Split-MNIST = {0,1}, {2,3}, ...). The class order is shuffled with a
  dedicated `class_order_seed` so that "which classes land together" is
  reproducible but not an artefact of the label numbering.

Swapping in a different dataset means writing one new entry in `_SPECS` and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import SEED, worker_init_fn


@dataclass(frozen=True)
class DatasetSpec:
    builder: str          # torchvision class name
    n_classes: int
    classes_per_task: int
    in_channels: int
    image_size: int
    mean: Sequence[float]
    std: Sequence[float]


_SPECS = {
    "split-mnist": DatasetSpec("MNIST", 10, 2, 1, 28, (0.1307,), (0.3081,)),
    "split-fashion": DatasetSpec("FashionMNIST", 10, 2, 1, 28, (0.2860,), (0.3530,)),
    "split-cifar10": DatasetSpec(
        "CIFAR10", 10, 2, 3, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
    "split-cifar100": DatasetSpec(
        "CIFAR100", 100, 10, 3, 32, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    ),
}


def available_datasets() -> List[str]:
    return sorted(_SPECS)


def get_spec(name: str) -> DatasetSpec:
    if name not in _SPECS:
        raise KeyError(f"unknown dataset {name!r}; choose from {available_datasets()}")
    return _SPECS[name]


def _transforms(spec: DatasetSpec, train: bool):
    ops = []
    if train and spec.in_channels == 3:
        # light augmentation only for the RGB datasets; MNIST-likes do not
        # need it and augmentation would muddy the forgetting signal
        ops += [
            transforms.RandomCrop(spec.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    return transforms.Compose(ops)


def _targets_array(ds) -> np.ndarray:
    t = ds.targets
    if isinstance(t, torch.Tensor):
        return t.numpy()
    return np.asarray(t)


@dataclass
class Task:
    index: int
    classes: List[int]          # original dataset labels belonging to this task
    train: Subset
    val: Subset
    test: Subset

    def __repr__(self) -> str:
        return (
            f"Task({self.index}, classes={self.classes}, "
            f"train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"
        )


class TaskSequence:
    """An ordered list of tasks plus the metadata the trainer needs."""

    def __init__(self, name: str, tasks: List[Task], spec: DatasetSpec, class_order: List[int]):
        self.name = name
        self.tasks = tasks
        self.spec = spec
        self.class_order = class_order

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, i: int) -> Task:
        return self.tasks[i]

    @property
    def n_classes(self) -> int:
        return self.spec.n_classes

    def class_mask(self, task_index: int, device=None) -> torch.Tensor:
        """Boolean mask over the full label space, True for this task's classes.

        Used for task-incremental evaluation: logits outside the mask are set
        to a large negative constant so the model only chooses among classes it
        is actually being asked about.
        """
        mask = torch.zeros(self.spec.n_classes, dtype=torch.bool)
        mask[self.tasks[task_index].classes] = True
        return mask.to(device) if device is not None else mask

    @property
    def class_to_task(self) -> np.ndarray:
        """`class_to_task[c]` = index of the task that owns class `c`.

        Every class belongs to exactly one task, so a sample's label is enough
        to recover which task it came from. That is what makes correct
        per-sample logit masking possible when a batch mixes tasks -- which
        happens with experience replay and with joint training. Unused class
        ids (possible if n_classes is not divisible by classes_per_task) map
        to -1.
        """
        c2t = np.full(self.spec.n_classes, -1, dtype=np.int64)
        for t in self.tasks:
            for c in t.classes:
                c2t[c] = t.index
        return c2t

    def task_class_mask_matrix(self) -> torch.Tensor:
        """Boolean `[n_tasks, n_classes]`; row t is True on task t's classes."""
        M = torch.zeros(len(self.tasks), self.spec.n_classes, dtype=torch.bool)
        for t in self.tasks:
            M[t.index, t.classes] = True
        return M


def build_task_sequence(
    dataset: str,
    data_root: str = "./data",
    val_fraction: float = 0.1,
    class_order_seed: int = SEED,
    download: bool = True,
) -> TaskSequence:
    spec = get_spec(dataset)
    cls = getattr(datasets, spec.builder)

    train_full = cls(root=data_root, train=True, download=download,
                     transform=_transforms(spec, train=True))
    # evaluation copy of the train set, without augmentation, for the val slice
    train_eval = cls(root=data_root, train=True, download=download,
                     transform=_transforms(spec, train=False))
    test_full = cls(root=data_root, train=False, download=download,
                    transform=_transforms(spec, train=False))

    y_train = _targets_array(train_full)
    y_test = _targets_array(test_full)

    rng = np.random.RandomState(class_order_seed)
    class_order = list(rng.permutation(spec.n_classes))

    n_tasks = spec.n_classes // spec.classes_per_task
    tasks: List[Task] = []

    for t in range(n_tasks):
        classes = [int(c) for c in class_order[t * spec.classes_per_task:(t + 1) * spec.classes_per_task]]

        train_idx = np.where(np.isin(y_train, classes))[0]
        test_idx = np.where(np.isin(y_test, classes))[0]

        # deterministic train/val carve-out, independent of run seed
        split_rng = np.random.RandomState(class_order_seed + 1000 + t)
        perm = split_rng.permutation(len(train_idx))
        n_val = int(round(val_fraction * len(train_idx)))
        val_sel = train_idx[perm[:n_val]]
        tr_sel = train_idx[perm[n_val:]]

        tasks.append(
            Task(
                index=t,
                classes=classes,
                train=Subset(train_full, tr_sel.tolist()),
                val=Subset(train_eval, val_sel.tolist()),
                test=Subset(test_full, test_idx.tolist()),
            )
        )

    return TaskSequence(dataset, tasks, spec, [int(c) for c in class_order])


def make_loader(subset, batch_size: int, shuffle: bool, num_workers: int = 2,
                seed: int = SEED, drop_last: bool = False,
                persistent: bool = True) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=gen if shuffle else None,
        drop_last=drop_last,
        persistent_workers=bool(num_workers) and persistent,
    )


def concat_train_subsets(seq: TaskSequence, upto: int = None):
    """All training data from tasks 0..upto, for the joint-training ceiling."""
    from torch.utils.data import ConcatDataset

    end = len(seq) if upto is None else upto + 1
    return ConcatDataset([seq[t].train for t in range(end)])
