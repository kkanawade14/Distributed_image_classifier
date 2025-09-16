"""Training entrypoint for multi-gpu CIFAR-10 benchmarks."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
yaml = None
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.data.cifar10 import create_dataloaders
from src.engine.evaluator import evaluate
from src.engine.trainer import Trainer
from src.models.build import build_model
from src.utils import dist as dist_utils
from src.utils.amp import get_grad_scaler
from src.utils.logging import BenchmarkLogger, TensorboardLogger, format_metrics
from src.utils.metrics import compute_scaling_efficiency
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 multi-GPU training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dist", choices=["none", "dp", "ddp"], default="none")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--workers", type=int, default=None, help="Number of dataloader workers")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--profiler", action="store_true", help="Enable PyTorch profiler and NVTX tracing")
    parser.add_argument("--baseline-time", type=float, default=None, help="Optional single-GPU baseline time for scaling efficiency")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Limit number of training batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Limit number of validation batches per epoch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--log-interval", type=int, default=None, help="Logging interval in steps")
    parser.add_argument("--output", type=str, default=None, help="Override checkpoint directory")
    return parser.parse_args()


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if yaml is None:
        raise ImportError("pyyaml is required to load configs")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    config: Dict[str, Any] = {}
    includes = data.pop("includes", []) or []
    for include in includes:
        include_path = (path.parent / include).resolve()
        config = _merge_dict(config, load_config(include_path))
    config = _merge_dict(config, data)
    return config


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    total_epochs: int,
    steps_per_epoch: int,
) -> lr_scheduler.LambdaLR | None:
    if not scheduler_cfg:
        return None
    name = scheduler_cfg.get("name", "cosine").lower()
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
    warmup_factor = scheduler_cfg.get("warmup_factor", 0.0)

    total_steps = max(1, total_epochs * max(1, steps_per_epoch))
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if name == "cosine":
        eta_min = scheduler_cfg.get("eta_min", 0.0)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_steps == 0:
                    return 1.0
                alpha = warmup_factor + (1 - warmup_factor) * (step / max(1, warmup_steps))
                return alpha
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(progress * math.pi))

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, filename: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / filename)


def resume_from_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    checkpoint_path: Path,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    best_acc = checkpoint.get("best_acc", 0.0)
    start_epoch = checkpoint.get("epoch", 0) + 1
    return start_epoch, best_acc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        config.setdefault("training", {})["grad_accum_steps"] = args.grad_accum_steps
    if args.log_interval is not None:
        config.setdefault("training", {})["log_interval"] = args.log_interval
    if args.output is not None:
        config.setdefault("training", {})["checkpoint_dir"] = args.output
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.workers is not None:
        config.setdefault("dataset", {})["num_workers"] = args.workers
    if args.limit_train_batches is not None:
        config.setdefault("training", {})["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config.setdefault("training", {})["limit_val_batches"] = args.limit_val_batches
    if args.resume:
        config.setdefault("training", {})["resume"] = args.resume

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})

    dist_utils.init_distributed_mode(args)

    seed_everything(training_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dist == "ddp" and torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank)
    print(f"Using device: {device}")

    per_device_batch = training_cfg.get("batch_size", 128)
    grad_accum = training_cfg.get("grad_accum_steps", 1)
    world_size = dist_utils.get_world_size() if args.dist == "ddp" else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    effective_batch = per_device_batch * grad_accum * max(1, world_size)
    print(
        f"Per-device batch size: {per_device_batch}, grad_accum_steps: {grad_accum}, "
        f"world_size: {world_size}, effective_batch_size: {effective_batch}"
    )

    train_loader, val_loader, train_sampler = create_dataloaders(
        config,
        dist_mode=args.dist,
        batch_size=per_device_batch,
        num_workers=dataset_cfg.get("num_workers"),
        limit_train_batches=training_cfg.get("limit_train_batches"),
        limit_val_batches=training_cfg.get("limit_val_batches"),
    )

    num_classes = dataset_cfg.get("num_classes", 10)
    model = build_model(model_cfg, num_classes=num_classes)
    model.to(device)

    if args.dist == "dp" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    elif args.dist == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if torch.cuda.is_available() else None,
            output_device=args.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    model_without_ddp = model.module if hasattr(model, "module") else model

    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    scheduler = create_scheduler(optimizer, scheduler_cfg, training_cfg.get("epochs", 1), steps_per_epoch)

    amp_enabled = bool(args.amp)
    scaler = get_grad_scaler(enabled=amp_enabled)

    label_smoothing = training_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    start_epoch = 0
    best_acc = 0.0
    resume_path = training_cfg.get("resume")
    if resume_path:
        start_epoch, best_acc = resume_from_checkpoint(
            model_without_ddp, optimizer, scheduler, scaler, Path(resume_path)
        )
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    enable_nvtx = args.profiler or os.getenv("ENABLE_NVTX", "0") == "1"
    enable_profiler = args.profiler or os.getenv("ENABLE_PROFILER", "0") == "1"

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        grad_accum_steps=grad_accum,
        amp_enabled=amp_enabled,
        log_interval=training_cfg.get("log_interval", 50),
        clip_grad_norm=training_cfg.get("clip_grad_norm"),
        enable_nvtx=enable_nvtx,
    )

    run_name = f"{model_cfg.get('name', 'model')}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    tb_logger = TensorboardLogger(args.logdir, run_name)
    benchmark_logger = BenchmarkLogger("benchmarks/results.csv") if dist_utils.is_main_process() else None

    epochs = training_cfg.get("epochs", 1)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))

    train_metrics: Dict[str, Any] = {}
    val_metrics: Dict[str, Any] = {}

    for epoch in range(start_epoch, epochs):
        if args.dist == "ddp" and isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
            train_sampler.set_epoch(epoch)

        profiler_cm = None
        if enable_profiler and epoch == start_epoch:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            trace_dir = Path("traces")
            trace_dir.mkdir(parents=True, exist_ok=True)
            profiler_cm = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=False,
                with_stack=False,
            )

        train_metrics = trainer.train_one_epoch(
            epoch,
            train_loader,
            criterion,
            train_sampler=train_sampler,
            limit_batches=training_cfg.get("limit_train_batches"),
            profiler=profiler_cm,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            limit_batches=training_cfg.get("limit_val_batches"),
        )

        if dist_utils.is_main_process():
            print(format_metrics(epoch, train_metrics, prefix="Train"))
            print(format_metrics(epoch, val_metrics, prefix="Val"))

            tb_logger.add_scalar("train/loss", train_metrics["loss"], epoch)
            tb_logger.add_scalar("train/top1", train_metrics["top1"], epoch)
            tb_logger.add_scalar("train/imgs_per_sec", train_metrics["imgs_per_sec"], epoch)
            tb_logger.add_scalar("val/loss", val_metrics["loss"], epoch)
            tb_logger.add_scalar("val/top1", val_metrics["top1"], epoch)

            is_best = val_metrics["top1"] > best_acc
            if is_best:
                best_acc = val_metrics["top1"]

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model_without_ddp.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "scaler_state": scaler.state_dict(),
                    "best_acc": best_acc,
                },
                checkpoint_dir,
                f"epoch_{epoch}.pth",
            )
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": model_without_ddp.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "scaler_state": scaler.state_dict(),
                        "best_acc": best_acc,
                    },
                    checkpoint_dir,
                    "best.pth",
                )

        dist_utils.barrier()

    if dist_utils.is_main_process() and train_metrics and val_metrics:
        row = {
            "date": datetime.utcnow().isoformat(),
            "gpus": max(1, world_size if args.dist == "ddp" else (torch.cuda.device_count() if torch.cuda.is_available() else 1)),
            "model": model_cfg.get("name", "model"),
            "imgs_per_sec": train_metrics["imgs_per_sec"],
            "epoch_time_s": train_metrics["epoch_time"],
            "top1": val_metrics["top1"],
            "amp": args.amp,
            "dist": args.dist,
        }
        if args.baseline_time and row["gpus"] > 1:
            row["scaling_efficiency"] = compute_scaling_efficiency(args.baseline_time, row["epoch_time_s"], row["gpus"])
        elif row["gpus"] == 1:
            row["scaling_efficiency"] = 100.0
        if benchmark_logger:
            benchmark_logger.append(row)

    tb_logger.close()
    dist_utils.cleanup_distributed()


if __name__ == "__main__":
    main()
