"""Unit tests for fmn.drift.select_layers -- the control-arm logic.

These matter more than they look. If `random` ever returns the same set as
`top`, or `matched` returns `top` itself, the control silently stops being a
control and the headline claim becomes unfalsifiable while still looking fine
in the plot.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmn.selection import (  # noqa: E402
    matched_param_budget,
    rank_layers_by_drift,
    select_layers,
)

# head deliberately has the largest drift: it must still never be selected
PROFILE = {
    "stem": 0.05,
    "layer1": 0.10,
    "layer2": 0.20,
    "layer3": 0.35,
    "layer4": 0.50,
    "head": 0.90,
}
PARAMS = {
    "stem": 600,
    "layer1": 15_000,
    "layer2": 60_000,
    "layer3": 240_000,
    "layer4": 950_000,
    "head": 1_600,
}


def test_ranking_excludes_head():
    ranked = rank_layers_by_drift(PROFILE)
    assert "head" not in ranked
    assert ranked == ["layer4", "layer3", "layer2", "layer1", "stem"]


def test_top_picks_highest_drift():
    assert select_layers(PROFILE, 2, "top") == ["layer4", "layer3"]


def test_bottom_picks_lowest_drift():
    assert set(select_layers(PROFILE, 2, "bottom")) == {"stem", "layer1"}


def test_bottom_never_overlaps_top_when_k_is_small():
    top = set(select_layers(PROFILE, 2, "top"))
    bottom = set(select_layers(PROFILE, 2, "bottom"))
    assert not (top & bottom)


def test_head_is_never_selected_in_any_mode():
    for mode in ["top", "bottom", "random", "paramrandom"]:
        for k in range(1, 5):
            chosen = select_layers(PROFILE, k, mode, param_counts=PARAMS, seed=7)
            assert "head" not in chosen, f"{mode} k={k} selected the head"


def test_random_is_seed_reproducible_and_seed_sensitive():
    a = select_layers(PROFILE, 2, "random", seed=42)
    b = select_layers(PROFILE, 2, "random", seed=42)
    assert a == b, "same seed must give the same control set"
    draws = {tuple(select_layers(PROFILE, 2, "random", seed=s)) for s in range(20)}
    assert len(draws) > 1, "random control never varies across seeds"


def test_random_returns_correct_count():
    for k in range(0, 6):
        assert len(select_layers(PROFILE, k, "random", seed=3)) == k


def test_paramrandom_freezes_no_layer_group():
    # the weight-level control deliberately freezes no whole layer; the
    # strategy builds a scattered weight mask instead
    for k in range(1, 5):
        assert select_layers(PROFILE, k, "paramrandom", param_counts=PARAMS) == []


def test_param_budget_matches_the_top_arm():
    for k in range(0, 6):
        top = select_layers(PROFILE, k, "top")
        assert matched_param_budget(PROFILE, k, PARAMS) == sum(PARAMS[g] for g in top)


def test_param_budget_never_counts_the_head():
    # head has the highest drift in PROFILE; if it leaked into the ranking the
    # budget would silently include it
    assert matched_param_budget(PROFILE, 5, PARAMS) == sum(
        PARAMS[g] for g in ["layer4", "layer3", "layer2", "layer1", "stem"])


def test_k_zero_freezes_nothing_in_every_mode():
    for mode in ["top", "bottom", "random", "paramrandom"]:
        assert select_layers(PROFILE, 0, mode, param_counts=PARAMS) == []


def test_k_is_clamped_to_available_layers():
    chosen = select_layers(PROFILE, 99, "top")
    assert len(chosen) == 5  # five candidates, head excluded


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_layers(PROFILE, 1, "sideways")
