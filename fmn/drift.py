"""Per-layer weight drift -- the diagnostic that the whole project hangs on.

The question: when a network forgets, *where* in the network does the damage
happen? We answer it by snapshotting the weights after every task and
measuring, per layer group, how far they moved.

Two drift measures are recorded:

* **relative L2 drift**  ||theta_t - theta_ref|| / ||theta_ref||
  Scale-free, so a 20-parameter bias vector and a 100k-parameter conv kernel
  are comparable. This is the headline number.
* **cosine drift**  1 - cos(theta_t, theta_ref)
  Catches *directional* change that L2 can miss when a layer is rescaled.

`ref` is the snapshot after the first task, so "drift" always means "movement
away from the state in which task 0 was solved". That is the state whose
destruction the forgetting metric is measuring, which keeps the diagnosis and
the outcome metric talking about the same thing.

BatchNorm running statistics are excluded: they are buffers, not parameters,
and they move for reasons unrelated to the weight interference we care about.
"""

from __future__ import annotations

import json
from typing import Dict, List

import torch


def snapshot(model) -> Dict[str, torch.Tensor]:
    """Detached CPU copy of all trainable parameters."""
    return {n: p.detach().clone().cpu() for n, p in model.named_parameters()}


def _group_vector(snap: Dict[str, torch.Tensor], names: List[str]) -> torch.Tensor:
    present = [snap[n].flatten() for n in names if n in snap]
    if not present:
        return torch.zeros(1)
    return torch.cat(present)


def layer_drift(
    ref: Dict[str, torch.Tensor],
    cur: Dict[str, torch.Tensor],
    groups: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Relative L2 and cosine drift for each layer group."""
    out: Dict[str, Dict[str, float]] = {}
    for gname, names in groups.items():
        a = _group_vector(ref, names)
        b = _group_vector(cur, names)
        denom = a.norm().item()
        rel_l2 = ((b - a).norm().item() / denom) if denom > 1e-12 else 0.0
        if denom > 1e-12 and b.norm().item() > 1e-12:
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        else:
            cos = 1.0
        out[gname] = {
            "rel_l2": float(rel_l2),
            "cosine_drift": float(1.0 - cos),
            "n_params": int(a.numel()),
        }
    return out


class DriftTracker:
    """Accumulates a drift profile across the task sequence."""

    def __init__(self, model):
        self.groups = model.layer_groups()
        self.ref: Dict[str, torch.Tensor] | None = None
        self.history: List[Dict[str, Dict[str, float]]] = []

    def set_reference(self, model) -> None:
        self.ref = snapshot(model)

    def record(self, model) -> Dict[str, Dict[str, float]]:
        if self.ref is None:
            raise RuntimeError("call set_reference() after the first task")
        d = layer_drift(self.ref, snapshot(model), self.groups)
        self.history.append(d)
        return d

    def profile(self) -> Dict[str, float]:
        """Mean relative-L2 drift per layer, averaged over recorded checkpoints.

        This single number per layer is what `drift_freeze` consumes.
        """
        if not self.history:
            return {}
        return {
            g: float(sum(h[g]["rel_l2"] for h in self.history) / len(self.history))
            for g in self.groups
        }

    def param_counts(self) -> Dict[str, int]:
        """Parameter count per layer group.

        Needed by the parameter-matched control: high-drift layers tend also
        to be the *large* layers, so "freezing helped" could just mean
        "freezing lots of parameters helped". Recording the counts lets a
        control arm match on parameter budget instead of layer count.
        """
        if not self.history:
            return {}
        return {g: int(self.history[0][g]["n_params"]) for g in self.groups}

    def to_dict(self) -> dict:
        return {
            "layer_order": list(self.groups),
            "per_checkpoint": self.history,
            "mean_rel_l2": self.profile(),
            "n_params": self.param_counts(),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# Re-exported for convenience: the selection policy lives in its own torch-free
# module so it can be unit-tested without PyTorch installed.
from .selection import (  # noqa: E402,F401
    load_param_counts,
    load_profile,
    matched_param_budget,
    rank_layers_by_drift,
    select_layers,
)
