"""Backbones.

Two models, both small enough that a full 5-task run finishes in minutes on
a laptop GPU:

* `SmallCNN`   -- 28x28 grayscale (MNIST / FashionMNIST)
* `ReducedResNet18` -- 32x32 RGB (CIFAR-10/100), the nf=20 variant used as
  the standard continual-learning backbone in Lopez-Paz & Ranzato (2017).

Both expose `layer_groups()`, which is what the drift analysis keys off.
Keeping that mapping inside the model (rather than string-matching parameter
names in the analysis code) means adding a new backbone does not silently
break the drift plots.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """~180k parameters. Trains a 2-class MNIST task in seconds."""

    def __init__(self, in_channels: int = 1, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.head = nn.Linear(128, n_classes)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.head(self.features(x))

    def layer_groups(self) -> Dict[str, List[str]]:
        return {
            "conv1": ["conv1.weight", "conv1.bias"],
            "conv2": ["conv2.weight", "conv2.bias"],
            "fc1": ["fc1.weight", "fc1.bias"],
            "head": ["head.weight", "head.bias"],
        }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ReducedResNet18(nn.Module):
    """ResNet-18 with 3x fewer filters (nf=20), the CL-standard CIFAR backbone."""

    def __init__(self, in_channels: int = 3, n_classes: int = 10, nf: int = 20):
        super().__init__()
        self.in_planes = nf
        self.conv1 = nn.Conv2d(in_channels, nf, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(nf, 2, 1)
        self.layer2 = self._make_layer(nf * 2, 2, 2)
        self.layer3 = self._make_layer(nf * 4, 2, 2)
        self.layer4 = self._make_layer(nf * 8, 2, 2)
        self.head = nn.Linear(nf * 8, n_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return torch.flatten(out, 1)

    def forward(self, x):
        return self.head(self.features(x))

    def layer_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {"stem": [], "layer1": [], "layer2": [],
                                        "layer3": [], "layer4": [], "head": []}
        for name, _ in self.named_parameters():
            if name.startswith("layer"):
                groups[name.split(".")[0]].append(name)
            elif name.startswith("head"):
                groups["head"].append(name)
            else:
                groups["stem"].append(name)
        return groups


def build_model(dataset_spec, n_classes: int) -> nn.Module:
    """Pick the backbone that matches the data shape."""
    if dataset_spec.in_channels == 1:
        return SmallCNN(dataset_spec.in_channels, n_classes)
    return ReducedResNet18(dataset_spec.in_channels, n_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
