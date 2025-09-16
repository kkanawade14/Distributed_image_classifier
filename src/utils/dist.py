"""Distributed training helpers."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)


def init_distributed_mode(args) -> None:
    """Initialize distributed training based on command-line args."""
    if args.dist != "ddp":
        args.rank = 0
        args.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        args.local_rank = 0
        setup_for_distributed(is_main_process=True)
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        LOGGER.warning("DDP requested but environment variables are missing; defaulting to rank 0")
        args.rank = 0
        args.world_size = 1
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    setup_for_distributed(is_main_process=(args.rank == 0))


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def setup_for_distributed(is_main_process: bool) -> None:
    """Disable printing for non-main process to reduce log spam."""

    import builtins as __builtins__  # noqa: N812

    builtin_print = __builtins__.print

    def print(*args, **kwargs):  # type: ignore[override]
        force = kwargs.pop("force", False)
        if is_main_process or force:
            builtin_print(*args, **kwargs)

    __builtins__.print = print  # type: ignore[attr-defined]


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    tensor = all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(get_world_size())
    return tensor


@contextmanager
def distributed_sync() -> None:
    try:
        yield
    finally:
        barrier()
