"""Transform builders for CIFAR-10."""
from __future__ import annotations

from typing import Any, Dict

import torchvision.transforms as T
from torchvision.transforms.autoaugment import AutoAugmentPolicy


def _build_normalize(cfg: Dict[str, Any]) -> T.Normalize:
    stats = cfg.get("normalization", {})
    mean = stats.get("mean", [0.4914, 0.4822, 0.4465])
    std = stats.get("std", [0.2470, 0.2435, 0.2616])
    return T.Normalize(mean=mean, std=std)


def build_train_transform(dataset_cfg: Dict[str, Any], augmentation_cfg: Dict[str, Any]) -> T.Compose:
    img_size = dataset_cfg.get("img_size", 32)
    aug = augmentation_cfg or {}
    ops = []

    resize = aug.get("resize")
    if resize:
        ops.append(T.Resize(resize))

    random_crop = aug.get("random_crop")
    if random_crop:
        ops.append(T.RandomCrop(random_crop.get("size", img_size), padding=random_crop.get("padding", 4)))
    elif resize:
        ops.append(T.RandomResizedCrop(resize, scale=(0.8, 1.0)))

    if aug.get("random_horizontal_flip", True):
        ops.append(T.RandomHorizontalFlip())

    color_jitter = aug.get("color_jitter")
    if color_jitter:
        ops.append(T.ColorJitter(**color_jitter))

    if aug.get("randaugment"):
        ops.append(T.RandAugment())

    if aug.get("autoaugment"):
        ops.append(T.AutoAugment(AutoAugmentPolicy.CIFAR10))

    ops.extend([T.ToTensor(), _build_normalize(aug)])
    return T.Compose(ops)


def build_eval_transform(dataset_cfg: Dict[str, Any], augmentation_cfg: Dict[str, Any]) -> T.Compose:
    img_size = dataset_cfg.get("img_size", 32)
    aug = augmentation_cfg or {}
    ops = []

    resize = aug.get("resize")
    if resize and resize != img_size:
        ops.append(T.Resize(resize))
        ops.append(T.CenterCrop(img_size))
    elif resize:
        ops.append(T.Resize(resize))
    ops.extend([T.ToTensor(), _build_normalize(aug)])
    return T.Compose(ops)
