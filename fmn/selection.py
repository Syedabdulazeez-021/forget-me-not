"""Layer-selection policy for the freezing experiments.

Kept free of torch on purpose: this is the logic that decides whether the
project has a falsifiable claim, so it must be unit-testable without a GPU,
a dataset, or even PyTorch installed.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List

__all__ = ["rank_layers_by_drift", "select_layers", "matched_param_budget",
           "parse_explicit_layers", "load_profile", "load_param_counts"]


def rank_layers_by_drift(profile: Dict[str, float], exclude: List[str] = None) -> List[str]:
    """Layer names sorted most-drifting first.

    `exclude` defaults to the classifier head. Freezing the head in a
    continual setting is degenerate -- new classes could never be learned --
    so it is never a freeze candidate, and saying so explicitly here stops
    that bug from ever appearing.
    """
    exclude = exclude or ["head"]
    items = [(g, v) for g, v in profile.items() if g not in exclude]
    return [g for g, _ in sorted(items, key=lambda kv: kv[1], reverse=True)]


def select_layers(
    profile: Dict[str, float],
    k: int,
    mode: str = "top",
    param_counts: Dict[str, int] = None,
    seed: int = 42,
    exclude: List[str] = None,
) -> List[str]:
    """Choose which layer groups to freeze.

    The non-`top` modes exist for falsification. "Freezing the highest-drift
    layers reduced forgetting" is not evidence that the drift measurement is
    informative -- freezing anything reduces plasticity, and in practice the
    high-drift layers are also the largest ones. Three controls separate those
    explanations:

      top         the method: highest mean relative drift
      bottom      opposite direction. If this works as well, the sign of the
                  drift signal carries no information.
      random      the same NUMBER OF LAYERS, chosen at random. If this ties
                  with `top`, the ranking adds nothing over freezing per se.
      paramrandom the same NUMBER OF WEIGHTS, scattered at random across the
                  whole network, ignoring layer boundaries. This is the
                  strictest control -- see `matched_param_budget` below. It
                  freezes no whole layer group, so this function returns an
                  empty list and the strategy builds a weight mask instead.

    A layer-level parameter-matched control was tried and abandoned: because
    the deepest layers hold the overwhelming majority of weights, no *disjoint*
    subset of layers comes close to the top-k budget, and the best match is
    always top-k plus a tiny layer -- which is the method again, not a control.
    `paramrandom` matches the budget exactly at weight granularity instead.

    `random` is seeded from the run seed, so different seeds genuinely
    resample the control rather than repeating one draw three times.
    """
    exclude = exclude or ["head"]
    ranked = rank_layers_by_drift(profile, exclude)
    k = max(0, min(k, len(ranked)))
    if k == 0 or mode == "paramrandom":
        return []

    if mode == "top":
        return ranked[:k]
    if mode == "bottom":
        return list(reversed(ranked))[:k]
    if mode == "random":
        rng = random.Random(seed)
        pool = list(ranked)
        rng.shuffle(pool)
        return sorted(pool[:k], key=ranked.index)

    raise ValueError(f"unknown freeze-select mode {mode!r}; "
                     "choose from top, bottom, random, paramrandom")


def parse_explicit_layers(
    spec: str,
    profile: Dict[str, float],
    exclude: List[str] = None,
) -> List[str]:
    """Parse a comma-separated layer list, e.g. "layer3" or "stem,layer1".

    Used by the forgetting-risk scan, which freezes one layer at a time to
    measure that layer's individual contribution to forgetting. Unknown names
    are rejected loudly: a typo would otherwise silently freeze nothing and
    produce a risk score of zero that looks like a real measurement.
    """
    exclude = exclude or ["head"]
    ranked = rank_layers_by_drift(profile, exclude)
    wanted = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [w for w in wanted if w not in profile]
    if unknown:
        raise ValueError(f"unknown layer(s) {unknown}; profile has {sorted(profile)}")
    blocked = [w for w in wanted if w in exclude]
    if blocked:
        raise ValueError(f"layer(s) {blocked} are excluded from freezing "
                         "(freezing the head makes new classes unlearnable)")
    return sorted(set(wanted), key=ranked.index)


def matched_param_budget(
    profile: Dict[str, float],
    k: int,
    param_counts: Dict[str, int],
    exclude: List[str] = None,
) -> int:
    """How many weights the `top` arm would freeze at this k.

    The `paramrandom` control freezes exactly this many individual weights,
    chosen at random across the network, so that "same number of weights
    frozen" is held constant while "which layers" is destroyed.
    """
    exclude = exclude or ["head"]
    ranked = rank_layers_by_drift(profile, exclude)
    k = max(0, min(k, len(ranked)))
    return int(sum(param_counts.get(g, 0) for g in ranked[:k]))


def load_profile(path: str) -> Dict[str, float]:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if "mean_rel_l2" not in blob:
        raise ValueError(f"{path} is not a drift profile (missing mean_rel_l2)")
    return blob["mean_rel_l2"]


def load_param_counts(path: str) -> Dict[str, int]:
    """Parameter counts per layer group, with a fallback for older profiles."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if blob.get("n_params"):
        return {g: int(v) for g, v in blob["n_params"].items()}
    per_cp = blob.get("per_checkpoint") or []
    if per_cp:
        return {g: int(d["n_params"]) for g, d in per_cp[0].items()}
    return {}
