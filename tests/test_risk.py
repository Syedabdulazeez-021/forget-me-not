"""Unit tests for fmn.risk -- the forgetting-risk and frontier analysis.

The correlation number in fig4 is the headline claim of the sub-problem, so
the statistics behind it get tested rather than trusted.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmn.risk import collect_risk, spearman, stability_plasticity  # noqa: E402


# ---------------------------------------------------------------- spearman
def test_spearman_perfect_positive():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    # strictly increasing but wildly non-linear -> still exactly 1.0
    assert spearman([1, 2, 3, 4], [1, 4, 900, 1000]) == pytest.approx(1.0)


def test_spearman_handles_ties():
    r = spearman([1, 1, 2, 3], [5, 5, 6, 7])
    assert r == pytest.approx(1.0)


def test_spearman_too_few_points_is_nan():
    assert np.isnan(spearman([1, 2], [3, 4]))


def test_spearman_constant_input_is_nan():
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


# --------------------------------------------------- stability / plasticity
def test_stability_excludes_the_final_task():
    res = {
        "diagonal_accuracy": [0.9, 0.9, 0.9],
        "final_per_task_accuracy": [0.2, 0.4, 0.95],
    }
    plasticity, stability = stability_plasticity(res)
    assert plasticity == pytest.approx(0.9)
    # the last task (0.95) must not inflate stability
    assert stability == pytest.approx(0.3)


def test_stability_plasticity_missing_fields_is_nan():
    p, s = stability_plasticity({})
    assert np.isnan(p) and np.isnan(s)


def test_single_task_run_has_no_stability():
    res = {"diagonal_accuracy": [0.8], "final_per_task_accuracy": [0.8]}
    p, s = stability_plasticity(res)
    assert p == pytest.approx(0.8)
    assert np.isnan(s)


# -------------------------------------------------------------- risk scores
def _run(method, forgetting, layers=None, mode=None, drift=None):
    r = {"method": method, "average_forgetting": forgetting}
    if layers is not None:
        r["method_info"] = {"select_mode": mode, "frozen_layers": layers,
                            "drift_profile": drift or {}}
    return r


DRIFT = {"stem": 0.05, "layer1": 0.20, "layer2": 0.50, "head": 0.90}


def test_risk_is_reduction_in_forgetting_versus_naive():
    runs = {
        "naive": _run("naive", 0.80),
        "a": _run("drift_freeze", 0.50, ["layer2"], "explicit", DRIFT),
        "b": _run("drift_freeze", 0.75, ["stem"], "explicit", DRIFT),
    }
    risk, drift, base = collect_risk(runs)
    assert base == pytest.approx(0.80)
    assert risk["layer2"] == pytest.approx(0.30)
    assert risk["stem"] == pytest.approx(0.05)
    assert drift == DRIFT


def test_risk_ignores_multi_layer_and_swept_runs():
    # only single-layer explicit runs isolate one layer's contribution
    runs = {
        "naive": _run("naive", 0.80),
        "swept": _run("drift_freeze", 0.40, ["layer2", "layer1"], "top", DRIFT),
        "multi": _run("drift_freeze", 0.30, ["layer2", "stem"], "explicit", DRIFT),
        "single": _run("drift_freeze", 0.60, ["layer1"], "explicit", DRIFT),
    }
    risk, _, _ = collect_risk(runs)
    assert set(risk) == {"layer1"}


def test_risk_averages_across_seeds():
    runs = {
        "naive": _run("naive", 0.80),
        "s1": _run("drift_freeze", 0.60, ["layer1"], "explicit", DRIFT),
        "s2": _run("drift_freeze", 0.40, ["layer1"], "explicit", DRIFT),
    }
    risk, _, _ = collect_risk(runs)
    assert risk["layer1"] == pytest.approx(0.30)


def test_risk_can_be_negative_when_freezing_hurts():
    # freezing a layer the model needed makes forgetting worse; the score must
    # be allowed to go negative rather than being clipped to zero
    runs = {
        "naive": _run("naive", 0.50),
        "a": _run("drift_freeze", 0.70, ["layer2"], "explicit", DRIFT),
    }
    risk, _, _ = collect_risk(runs)
    assert risk["layer2"] == pytest.approx(-0.20)


def test_no_naive_run_yields_no_risk():
    runs = {"a": _run("drift_freeze", 0.50, ["layer2"], "explicit", DRIFT)}
    risk, _, base = collect_risk(runs)
    assert risk == {}
    assert np.isnan(base)
