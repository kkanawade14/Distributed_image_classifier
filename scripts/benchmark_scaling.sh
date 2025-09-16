#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/cifar10_resnet50.yaml}
if [[ $# -gt 0 ]]; then
  shift || true
fi
EXTRA_ARGS=("$@")

run_single() {
  echo "[benchmark] Running single GPU baseline"
  python -m src.train --config "${CONFIG}" --dist none --grad-accum-steps 4 --amp "${EXTRA_ARGS[@]}"
}

run_ddp() {
  local gpus=$1
  local accum=$2
  shift 2 || true
  echo "[benchmark] Running DDP with ${gpus} GPUs (grad_accum_steps=${accum})"
  bash "$(dirname "$0")/launch_ddp.sh" "${gpus}" --config "${CONFIG}" --grad-accum-steps "${accum}" --amp "${EXTRA_ARGS[@]}"
}

run_single
run_ddp 2 2
run_ddp 4 1

echo "[benchmark] Results appended to benchmarks/results.csv"
