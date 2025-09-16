.PHONY: setup train_ddp train_dp train_single bench test

setup:
@echo "Install dependencies via pip install -r requirements.txt or conda env create -f environment.yml"

train_single:
python -m src.train --config configs/cifar10_resnet50.yaml --dist none --amp

train_dp:
python -m src.train --config configs/cifar10_resnet50.yaml --dist dp --amp

train_ddp:
bash scripts/launch_ddp.sh 4 --config configs/cifar10_resnet50.yaml --amp

bench:
bash scripts/benchmark_scaling.sh

test:
python -m pytest -q
