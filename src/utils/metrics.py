"""Metric utilities for training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

from . import dist as dist_utils


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Iterable[torch.Tensor]:
    """Compute the top-k accuracy for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@dataclass
class AverageMeter:
    """Track running averages."""

    value: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def synchronize_between_processes(self) -> None:
        if not dist_utils.is_dist_avail_and_initialized():
            return
        tensor = torch.tensor([self.sum, self.count], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dist_utils.all_reduce(tensor)
        self.sum = tensor[0].item()
        self.count = int(tensor[1].item())
        if self.count > 0:
            self.avg = self.sum / self.count


def compute_throughput(num_samples: int, duration_s: float) -> float:
    """Return the images processed per second."""
    if duration_s <= 0:
        return 0.0
    return num_samples / duration_s


def compute_scaling_efficiency(t1: float, tn: float, n: int) -> float:
    """Compute scaling efficiency given single GPU and N-GPU timings."""
    if tn <= 0 or n <= 0:
        return 0.0
    return (t1 / (n * tn)) * 100.0
