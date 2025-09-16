"""Logging helpers for training and benchmarking."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from . import dist as dist_utils
from .metrics import compute_scaling_efficiency


class TensorboardLogger:
    """Wrapper around SummaryWriter that only logs on the main process."""

    def __init__(self, log_dir: str, run_name: str) -> None:
        self._writer: Optional[SummaryWriter]
        if dist_utils.is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
        else:
            self._writer = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


class BenchmarkLogger:
    """Append benchmark rows to a CSV file and compute scaling efficiency."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        if self.csv_path.suffix != ".csv":
            raise ValueError("csv_path must point to a CSV file")
        if not self.csv_path.parent.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "date",
                    "gpus",
                    "model",
                    "imgs_per_sec",
                    "epoch_time_s",
                    "top1",
                    "amp",
                    "dist",
                    "scaling_efficiency",
                ])
                writer.writeheader()

    def _lookup_baseline(self, model: str) -> Optional[float]:
        if not self.csv_path.exists():
            return None
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") == model and row.get("gpus") == "1":
                    try:
                        return float(row.get("epoch_time_s", 0.0))
                    except (TypeError, ValueError):
                        continue
        return None

    def append(self, row: Dict[str, object]) -> None:
        row = dict(row)
        model = str(row.get("model", ""))
        gpus = int(row.get("gpus", 1))
        epoch_time = float(row.get("epoch_time_s", 0.0))
        if gpus > 1 and ("scaling_efficiency" not in row or row["scaling_efficiency"] in (None, "")):
            baseline = self._lookup_baseline(model)
            if baseline:
                row["scaling_efficiency"] = compute_scaling_efficiency(baseline, epoch_time, gpus)
        if "scaling_efficiency" not in row or row["scaling_efficiency"] in (None, ""):
            row["scaling_efficiency"] = ""
        row.setdefault("date", datetime.utcnow().isoformat())
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date",
                "gpus",
                "model",
                "imgs_per_sec",
                "epoch_time_s",
                "top1",
                "amp",
                "dist",
                "scaling_efficiency",
            ])
            writer.writerow(row)


def format_metrics(epoch: int, metrics: Dict[str, float], prefix: str) -> str:
    """Create a human-readable metrics string."""
    parts = [f"{prefix} Epoch {epoch}"]
    for key, value in metrics.items():
        parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)
