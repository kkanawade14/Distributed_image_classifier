"""Generate speedup and scaling efficiency plots from benchmark CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speedup"] = float("nan")
    baselines = df[(df["gpus"] == 1) & (df["dist"] == "none")]
    for (model,), baseline_group in baselines.groupby("model"):
        if baseline_group.empty:
            continue
        t1 = baseline_group.iloc[-1]["epoch_time_s"]
        mask = df["model"] == model
        df.loc[mask, "speedup"] = t1 / df.loc[mask, "epoch_time_s"].replace({0: float("nan")})
        df.loc[mask, "scaling_efficiency"] = (
            t1 / (df.loc[mask, "gpus"].replace({0: float("nan")}) * df.loc[mask, "epoch_time_s"].replace({0: float("nan")}))
        ) * 100.0
    return df


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for (model, dist), group in df.groupby(["model", "dist"]):
        group = group.sort_values("gpus")
        plt.plot(group["gpus"], group[metric], marker="o", label=f"{model} ({dist})")
    plt.xlabel("GPUs")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark speedup and scaling efficiency")
    parser.add_argument("--csv", type=str, default="benchmarks/results.csv")
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Benchmark CSV is empty")

    df = compute_speedup(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(df, "speedup", "Speedup", output_dir / "speedup.png")
    if "scaling_efficiency" in df.columns:
        plot_metric(df, "scaling_efficiency", "Scaling Efficiency (%)", output_dir / "scaling_efficiency.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
