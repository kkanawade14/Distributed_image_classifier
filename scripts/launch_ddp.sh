#!/usr/bin/env bash
set -euo pipefail
NPROC=${1:-4}
if [[ $# -gt 0 ]]; then
  shift
fi
torchrun --standalone --nproc_per_node=${NPROC} src/train.py --dist ddp "$@"
