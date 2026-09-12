"""Unit tests for fmn.metrics. Run with: python -m pytest tests -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmn.metrics import (  # noqa: E402
    average_accuracy,
    average_forgetting,
    backward_transfer,
    summarize,
)


def test_perfect_memory_has_no_forgetting():
    # model keeps every task at 0.9 forever
    R = np.full((4, 4), 0.9)
    assert average_accuracy(R) == pytest.approx(0.9)
    assert average_forgetting(R) == pytest.approx(0.0)
    assert backward_transfer(R) == pytest.approx(0.0)


def test_total_forgetting():
    # each task is learned to 1.0 then collapses to 0.0 immediately after
    T = 3
    R = np.zeros((T, T))
    for i in range(T):
        R[i, i] = 1.0
    assert average_accuracy(R) == pytest.approx(1.0 / 3)
    # tasks 0 and 1 each drop from 1.0 to 0.0
    assert average_forgetting(R) == pytest.approx(1.0)
    assert backward_transfer(R) == pytest.approx(-1.0)


def test_forgetting_excludes_final_task():
    # two tasks: task 0 drops 1.0 -> 0.4, task 1 just learned
    R = np.array([[1.0, 0.1], [0.4, 0.95]])
    assert average_forgetting(R) == pytest.approx(0.6)
    assert average_accuracy(R) == pytest.approx((0.4 + 0.95) / 2)


def test_forgetting_uses_best_not_first():
    # task 0 dips then recovers then drops; best-ever before the end is 0.9
    R = np.array(
        [
            [0.8, 0.0, 0.0],
            [0.9, 0.7, 0.0],
            [0.5, 0.6, 0.8],
        ]
    )
    # task 0: best over rows 0..1 = 0.9, final 0.5 -> drop 0.4
    # task 1: best over rows 1..1 = 0.7, final 0.6 -> drop 0.1
    assert average_forgetting(R) == pytest.approx(0.25)


def test_positive_backward_transfer():
    # later task improves the earlier one
    R = np.array([[0.7, 0.0], [0.8, 0.9]])
    assert backward_transfer(R) == pytest.approx(0.1)
    assert average_forgetting(R) == pytest.approx(-0.1)


def test_single_task_is_degenerate_but_safe():
    R = np.array([[0.88]])
    assert average_accuracy(R) == pytest.approx(0.88)
    assert average_forgetting(R) == 0.0
    assert backward_transfer(R) == 0.0


def test_summarize_is_json_shaped():
    R = np.array([[0.9, 0.1], [0.5, 0.9]])
    out = summarize(R)
    assert set(out) == {
        "average_accuracy",
        "average_forgetting",
        "backward_transfer",
        "forward_transfer",
        "final_per_task_accuracy",
        "diagonal_accuracy",
        "n_tasks",
    }
    assert out["n_tasks"] == 2
    assert all(isinstance(v, float) for v in out["final_per_task_accuracy"])


def test_rejects_non_square():
    with pytest.raises(ValueError):
        average_accuracy(np.zeros((2, 3)))
