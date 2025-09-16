# multi-gpu-cifar10

`multi-gpu-cifar10` is a reference PyTorch project that showcases efficient CIFAR-10
training on single GPUs, `nn.DataParallel`, and `DistributedDataParallel` (DDP) with
automatic mixed precision (AMP). The repository provides production-ready training
and evaluation scripts, configuration-driven experiments, benchmark automation, and
reporting utilities to study scaling efficiency across 1→2→4 GPUs.

## Features

* ResNet-50 and ViT-Tiny architectures (torchvision + timm).
* Turn-key support for `--dist {none, dp, ddp}` to switch between execution modes.
* Automatic mixed precision via `--amp` and gradient accumulation for effective batch
  size parity across GPU counts.
* YAML-based experiment configuration for data, augmentation, optimization, and
  scheduling.
* Benchmarks that log images/sec, epoch time, top-1 accuracy, and scaling efficiency
  to CSV with plotting utilities.
* Scripts for launching DDP, DataParallel, profiling with NVTX/profiler, and sweeping
  benchmark runs.

## Repository structure

```
.
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ configs/
│  ├─ cifar10_resnet50.yaml
│  ├─ cifar10_vit_tiny.yaml
│  ├─ aug_base.yaml
│  └─ optim_adamw_cosine.yaml
├─ scripts/
│  ├─ launch_ddp.sh
│  ├─ launch_dp.sh
│  ├─ benchmark_scaling.sh
│  └─ profile_nvtx.sh
├─ src/
│  ├─ train.py
│  ├─ eval.py
│  ├─ engine/
│  │  ├─ trainer.py
│  │  └─ evaluator.py
│  ├─ data/
│  │  ├─ cifar10.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ build.py
│  │  └─ timm_wrappers.py
│  └─ utils/
│     ├─ dist.py
│     ├─ metrics.py
│     ├─ logging.py
│     ├─ amp.py
│     └─ seed.py
├─ tests/
│  ├─ test_dataloader.py
│  └─ test_smoke_train.py
└─ benchmarks/
   ├─ results.csv
   └─ plots.py
```

## Setup

1. (Optional) Create a conda environment: `conda env create -f environment.yml`
2. Alternatively, install dependencies into an existing environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure CUDA-capable GPUs are visible for multi-GPU benchmarks. CPU-only
   execution is supported for functionality checks.

## Quickstart

The commands below assume execution from the repository root and that CIFAR-10 is
not yet downloaded (the scripts will download automatically).

```bash
# Single GPU / CPU training
python -m src.train --config configs/cifar10_resnet50.yaml --dist none --amp

# DataParallel across visible GPUs
python -m src.train --config configs/cifar10_resnet50.yaml --dist dp --amp

# DistributedDataParallel with torchrun (4 GPUs)
bash scripts/launch_ddp.sh 4 --config configs/cifar10_resnet50.yaml --amp

# Benchmark 1→2→4 GPU sweep (DDP)
bash scripts/benchmark_scaling.sh
```

Use `--model` to override the configuration model (e.g., `vit_tiny_patch16_224`) and
`--grad-accum-steps` to preserve effective batch sizes when changing GPU counts.
The scripts automatically communicate the effective batch size and per-device batch
size at runtime.

## Expected metrics & scaling efficiency

Example numbers (representative on A100 GPUs):

| GPUs | Dist | Model  | imgs/sec | epoch time (s) | top-1 (%) | scaling eff. |
|------|------|--------|----------|----------------|-----------|---------------|
| 1    | none | ResNet | ~3200    | ~16            | 93.4      | 100           |
| 2    | ddp  | ResNet | ~6100    | ~8.5           | 93.4      | ~94           |
| 4    | ddp  | ResNet | ~11800   | ~4.3           | 93.3      | ~93           |

Scaling efficiency is computed as:

```
scaling_efficiency = (T1 / (N * TN)) * 100
```

where `T1` is the epoch time for the single-GPU baseline, `N` is the GPU count, and
`TN` is the epoch time for the N-GPU run. The benchmark script appends each run to
`benchmarks/results.csv` and automatically computes efficiency whenever a matching
single-GPU baseline exists.

## Benchmarking workflow

1. Train a single-GPU baseline (`--dist none`) to populate the benchmark CSV.
2. Run DataParallel or DDP variants. Metrics are logged to TensorBoard (under
   `runs/`) and to `benchmarks/results.csv`.
3. Generate scaling plots:

   ```bash
   python benchmarks/plots.py
   ```

   This produces speedup and scaling efficiency PNGs next to the CSV.

## Profiling

To record NVTX ranges and a PyTorch profiler trace, use:

```bash
bash scripts/profile_nvtx.sh --config configs/cifar10_resnet50.yaml --amp
```

The trace is written to `./traces/` for inspection via Chrome Trace Viewer or
TensorBoard.

## Testing

Execute the smoke tests and dataloader checks with:

```bash
make test
```

These tests instantiate data pipelines and run a short, CPU-only training loop.

---

Happy benchmarking and may your scaling curves be steep!
