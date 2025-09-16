"""Model factory."""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from torchvision import models as tv_models

from . import timm_wrappers


def _adapt_resnet_for_cifar(model: nn.Module) -> nn.Module:
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()
    return model


def build_model(config: Dict[str, Any], num_classes: int = 10) -> nn.Module:
    name = config.get("name", "resnet50").lower()
    pretrained = config.get("pretrained", False)

    if name == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        model = _adapt_resnet_for_cifar(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vit_tiny_patch16_224":
        model = timm_wrappers.create_vit(name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model
