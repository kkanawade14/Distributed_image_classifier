"""Model evaluation loop."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from src.utils.metrics import AverageMeter, topk_accuracy


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            if limit_batches is not None and step >= limit_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc1 = topk_accuracy(outputs, targets, topk=(1,))[0]
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))

    loss_meter.synchronize_between_processes()
    acc1_meter.synchronize_between_processes()

    return {"loss": loss_meter.avg, "top1": acc1_meter.avg}
