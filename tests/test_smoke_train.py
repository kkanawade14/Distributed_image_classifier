from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_smoke_train():
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        "configs/cifar10_resnet50.yaml",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--grad-accum-steps",
        "1",
        "--limit-train-batches",
        "1",
        "--limit-val-batches",
        "1",
        "--dist",
        "none",
        "--log-interval",
        "1",
        "--workers",
        "0",
    ]
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])
