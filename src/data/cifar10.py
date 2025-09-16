"""CIFAR-10 data module."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.datasets import CIFAR10

from .transforms import build_eval_transform, build_train_transform
from src.utils import dist as dist_utils
from src.utils.seed import worker_init_fn


def _build_subset(dataset, limit_batches: int, batch_size: int) -> Subset:
    subset_len = min(len(dataset), limit_batches * batch_size)
    indices = list(range(subset_len))
    return Subset(dataset, indices)


def create_dataloaders(
    config: Dict[str, Any],
    dist_mode: str = "none",
    batch_size: int | None = None,
    num_workers: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> Tuple[DataLoader, DataLoader, DistributedSampler | None]:
    dataset_cfg = config.get("dataset", {})
    aug_cfg = config.get("augmentation", {})
    training_cfg = config.get("training", {})

    per_device_batch = batch_size or training_cfg.get("batch_size", 128)
    workers = num_workers if num_workers is not None else dataset_cfg.get("num_workers", 4)
    pin_memory = dataset_cfg.get("pin_memory", True)
    deterministic_workers = dataset_cfg.get("deterministic_workers", False)

    root = dataset_cfg.get("data_dir", "./data")
    download = dataset_cfg.get("download", True)

    train_dataset = CIFAR10(root=root, train=True, download=download, transform=build_train_transform(dataset_cfg, aug_cfg))
    val_dataset = CIFAR10(root=root, train=False, download=download, transform=build_eval_transform(dataset_cfg, aug_cfg))

    if limit_train_batches:
        train_dataset = _build_subset(train_dataset, limit_train_batches, per_device_batch)
    if limit_val_batches:
        val_dataset = _build_subset(val_dataset, limit_val_batches, per_device_batch)

    train_sampler = None
    val_sampler = None

    if dist_mode == "ddp" and dist_utils.is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    persistent_workers = workers > 0
    loader_kwargs = {
        "batch_size": per_device_batch,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": worker_init_fn if deterministic_workers else None,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, train_sampler
