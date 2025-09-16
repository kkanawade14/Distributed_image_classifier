#!/usr/bin/env bash
set -euo pipefail
export ENABLE_NVTX=1
export ENABLE_PROFILER=1
python -m src.train "$@" --profiler
