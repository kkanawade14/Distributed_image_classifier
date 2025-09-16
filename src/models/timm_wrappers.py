"""Thin wrappers around timm model creation."""
from __future__ import annotations

from typing import Any

import timm


def create_vit(model_name: str, *, pretrained: bool, num_classes: int, **kwargs: Any):
    """Create a Vision Transformer using timm."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
