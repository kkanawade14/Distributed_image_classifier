#!/usr/bin/env bash
set -euo pipefail
python src/train.py --dist dp "$@"
