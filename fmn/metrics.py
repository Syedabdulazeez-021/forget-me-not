"""Continual-learning metrics.

Deliberately pure-NumPy (no torch) so the scoring logic is unit-testable
and cannot silently break when the training code changes.

The central object is the *accuracy matrix* R, where

    R[i][j] = accuracy on task j after finishing training on task i

R is lower-triangular-ish in meaning: entries with j > i are accuracies on
tasks the model has not seen yet. We still record them (they are cheap and
occasionally interesting for forward transfer) but no headline metric uses
them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_accuracy",
    "average_forgetting",
    "backward_transfer",
    "forward_transfer",
    "summarize",
]


def _as_matrix(R) -> np.ndarray:
    A = np.asarray(R, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"accuracy matrix must be square, got shape {A.shape}")
    return A


def average_accuracy(R) -> float:
    """Mean accuracy over all tasks, measured after the final task.

    This is the primary headline number required by the track brief.
    """
    A = _as_matrix(R)
    return float(A[-1, :].mean())


def average_forgetting(R) -> float:
    """Mean forgetting over all tasks except the last.

    Forgetting on task j is the drop from the best accuracy ever recorded
    on task j (over checkpoints taken after tasks j..T-2) to its accuracy at
    the very end. Defined this way in Chaudhry et al., 2018 (RWalk).

    The last task is excluded because it has had no opportunity to be
    forgotten -- including it would dilute the metric toward zero and make
    every method look better than it is.
    """
    A = _as_matrix(R)
    T = A.shape[0]
    if T < 2:
        return 0.0
    drops = []
    for j in range(T - 1):
        # checkpoints after task j up to (but excluding) the final one
        best_before_end = A[j:T - 1, j].max()
        drops.append(best_before_end - A[T - 1, j])
    return float(np.mean(drops))


def backward_transfer(R) -> float:
    """BWT: how much learning later tasks changed earlier-task accuracy.

    Negative BWT is forgetting; positive BWT means later tasks *helped*
    earlier ones. Defined in Lopez-Paz & Ranzato, 2017 (GEM).
    """
    A = _as_matrix(R)
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, j] - A[j, j] for j in range(T - 1)]))


def forward_transfer(R, random_baseline=None) -> float:
    """FWT: accuracy on task j before training on it, vs a random-init model.

    `random_baseline[j]` is the accuracy of an untrained model on task j.
    If omitted, 1/n_classes is assumed to be supplied by the caller instead;
    passing None returns the raw pre-training accuracies' mean.
    """
    A = _as_matrix(R)
    T = A.shape[0]
    if T < 2:
        return 0.0
    pre = [A[j - 1, j] for j in range(1, T)]
    if random_baseline is None:
        return float(np.mean(pre))
    b = np.asarray(random_baseline, dtype=float)
    return float(np.mean([pre[j - 1] - b[j] for j in range(1, T)]))


def summarize(R, random_baseline=None) -> dict:
    """All headline numbers in one dict, ready to dump to JSON."""
    A = _as_matrix(R)
    return {
        "average_accuracy": average_accuracy(A),
        "average_forgetting": average_forgetting(A),
        "backward_transfer": backward_transfer(A),
        "forward_transfer": forward_transfer(A, random_baseline),
        "final_per_task_accuracy": [float(x) for x in A[-1, :]],
        "diagonal_accuracy": [float(A[i, i]) for i in range(A.shape[0])],
        "n_tasks": int(A.shape[0]),
    }
