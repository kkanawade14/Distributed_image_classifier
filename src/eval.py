"""Evaluation entrypoint for CIFAR-10 models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.cifar10 import create_dataloaders
from src.engine.evaluator import evaluate
from src.models.build import build_model
from src.train import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.workers is not None:
        config.setdefault("dataset", {})["num_workers"] = args.workers
    if args.limit_val_batches is not None:
        config.setdefault("training", {})["limit_val_batches"] = args.limit_val_batches

    training_cfg: Dict[str, Any] = config.get("training", {})
    dataset_cfg: Dict[str, Any] = config.get("dataset", {})
    model_cfg: Dict[str, Any] = config.get("model", {})

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = create_dataloaders(
        config,
        dist_mode="none",
        batch_size=training_cfg.get("batch_size", 128),
        num_workers=dataset_cfg.get("num_workers"),
        limit_val_batches=training_cfg.get("limit_val_batches"),
    )

    num_classes = dataset_cfg.get("num_classes", 10)
    model = build_model(model_cfg, num_classes=num_classes)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    metrics = evaluate(
        model,
        val_loader,
        device,
        amp_enabled=args.amp,
        limit_batches=training_cfg.get("limit_val_batches"),
    )
    print(f"Eval loss: {metrics['loss']:.4f}, top-1: {metrics['top1']:.2f}")


if __name__ == "__main__":
    main()
