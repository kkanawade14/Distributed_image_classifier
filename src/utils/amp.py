"""Helpers for automatic mixed precision training."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def get_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    """Create a GradScaler configured for AMP."""
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def autocast(enabled: bool) -> Iterator[None]:
    """Context manager proxy for torch.cuda.amp.autocast."""
    with torch.cuda.amp.autocast(enabled=enabled):
        yield
