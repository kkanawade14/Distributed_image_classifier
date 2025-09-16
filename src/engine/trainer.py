"""Training loop implementation."""
from __future__ import annotations

import contextlib
import time
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

try:
    import torch.cuda.nvtx as nvtx
except ImportError:  # pragma: no cover - CPU-only environments
    nvtx = None

from src.utils import dist as dist_utils
from src.utils.amp import autocast
from src.utils.metrics import AverageMeter, compute_throughput, topk_accuracy


class Trainer:
    """Encapsulates the training loop for one epoch."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        *,
        grad_accum_steps: int = 1,
        amp_enabled: bool = False,
        log_interval: int = 50,
        clip_grad_norm: Optional[float] = None,
        enable_nvtx: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.amp_enabled = amp_enabled
        self.log_interval = log_interval
        self.clip_grad_norm = clip_grad_norm
        self.enable_nvtx = enable_nvtx and torch.cuda.is_available()
        self.global_step = 0

    def train_one_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        train_sampler: Optional[torch.utils.data.distributed.DistributedSampler] = None,
        limit_batches: Optional[int] = None,
        profiler: Optional[object] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        num_samples = 0

        start_time = time.time()
        profiler_cm = profiler if profiler is not None else contextlib.nullcontext()

        with profiler_cm:
            data_iter = enumerate(train_loader)
            if dist_utils.is_main_process():
                data_iter = tqdm(data_iter, total=len(train_loader), desc=f"Epoch {epoch}", leave=False)

            for step, (images, targets) in data_iter:
                if limit_batches is not None and step >= limit_batches:
                    break
                if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                    # Sampler epoch handled outside but ensure deterministic behaviour when limit batches used
                    pass

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.enable_nvtx and nvtx is not None:
                    nvtx.range_push(f"train_step_{step}")

                with autocast(self.amp_enabled):
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    loss_to_scale = loss / self.grad_accum_steps

                self.scaler.scale(loss_to_scale).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

                acc1 = topk_accuracy(outputs, targets, topk=(1,))[0]
                loss_meter.update(loss.item(), images.size(0))
                acc1_meter.update(acc1.item(), images.size(0))

                if dist_utils.is_main_process() and self.log_interval and (step + 1) % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tqdm.write(
                        f"Epoch {epoch} Step {step + 1}/{len(train_loader)} "
                        f"loss={loss_meter.avg:.4f} acc1={acc1_meter.avg:.2f} lr={current_lr:.6f}"
                    )

                if profiler is not None:
                    profiler.step()

                if self.enable_nvtx and nvtx is not None:
                    nvtx.range_pop()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time

        loss_meter.synchronize_between_processes()
        acc1_meter.synchronize_between_processes()

        metrics = {
            "loss": loss_meter.avg,
            "top1": acc1_meter.avg,
            "epoch_time": epoch_time,
            "imgs_per_sec": compute_throughput(len(train_loader.dataset), epoch_time),
        }
        return metrics
