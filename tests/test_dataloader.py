from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.cifar10 import create_dataloaders
from src.train import load_config


def test_dataloader_shapes(tmp_path):
    config = load_config("configs/cifar10_resnet50.yaml")
    config.setdefault("training", {})["batch_size"] = 8
    config.setdefault("training", {})["limit_train_batches"] = 1
    config.setdefault("training", {})["limit_val_batches"] = 1
    config.setdefault("dataset", {})["num_workers"] = 0

    train_loader, val_loader, _ = create_dataloaders(
        config,
        dist_mode="none",
        batch_size=config["training"]["batch_size"],
        limit_train_batches=config["training"]["limit_train_batches"],
        limit_val_batches=config["training"]["limit_val_batches"],
    )

    images, targets = next(iter(train_loader))
    assert images.shape[0] == config["training"]["batch_size"]
    assert images.shape[1:] == (3, 32, 32)
    assert targets.shape[0] == config["training"]["batch_size"]

    val_images, val_targets = next(iter(val_loader))
    assert val_images.shape[1:] == (3, 32, 32)
    assert val_targets.shape[0] == config["training"]["batch_size"]
    assert torch.all((targets >= 0) & (targets < 10))
