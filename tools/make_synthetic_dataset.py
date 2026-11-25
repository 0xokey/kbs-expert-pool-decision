"""Generate a controlled synthetic multivariate series for routing studies.

Each channel follows a different generating process, so the *best expert* differs
across channels by construction.  This gives a positive control for the routing
mechanism: if channel-wise routing cannot exploit such heterogeneity, it cannot
exploit anything.

    python tools/make_synthetic_dataset.py --out csv_data/synthetic_hetero.csv --rows 8000
    python tools/make_synthetic_dataset.py --out csv_data/synthetic_homo.csv --rows 8000 --mode homogeneous
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ar1(n, phi=0.9, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


def make_series(rows: int, mode: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    channels = {}
    if mode == "balanced":
        # Four process families, two channels each, equal difficulty by design:
        # a linear model should own the smooth family, a conv/UNet the noisy
        # local one, a recurrent model the long-memory one, and a transformer
        # the nonlinear one.
        smooth = 1.0 * np.sin(2 * np.pi * t / 24.0) + 0.02 * rng.normal(0, 1, rows)
        smooth2 = 0.8 * np.sin(2 * np.pi * t / 12.0) + 0.18 * np.sin(2 * np.pi * t / 24.0)
        channels["smooth_a"] = smooth
        channels["smooth_b"] = smooth2 + 0.02 * rng.normal(0, 1, rows)
        noisy_a = np.sin(2 * np.pi * t / 24.0) + 1.9 * _ar1(rows, 0.5, 0.25, seed=11)
        noisy_b = np.sin(2 * np.pi * t / 12.0) + 1.6 * rng.normal(0, 0.3, rows)
        channels["noisy_a"] = noisy_a
        channels["noisy_b"] = noisy_b
        channels["memory_a"] = _ar1(rows, 0.97, 0.2, seed=21)
        channels["memory_b"] = _ar1(rows, 0.9, 0.25, seed=22) + 0.2 * _ar1(rows, 0.7, 0.2, seed=23)
        square = np.sign(np.sin(2 * np.pi * t / 24.0)) * 0.9
        saw = 2.0 * ((t % 24) / 24.0) - 1.0
        channels["nonlinear_a"] = square + 0.05 * rng.normal(0, 1, rows)
        channels["nonlinear_b"] = saw + 0.05 * rng.normal(0, 1, rows)
    elif mode == "homogeneous":
        base = np.sin(2 * np.pi * t / 24.0)
        for c in range(6):
            channels[f"ch{c}"] = base + rng.normal(0, 0.1, rows)
    else:
        channels["seasonal"] = np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.05, rows)
        channels["seasonal_ar"] = np.sin(2 * np.pi * t / 24.0) + _ar1(rows, 0.85, 0.2, seed=1)
        channels["ar2"] = _ar1(rows, 0.95, 0.3, seed=2) + 0.3 * _ar1(rows, 0.6, 0.3, seed=3)
        channels["trend"] = 0.0004 * t + rng.normal(0, 0.15, rows)
        channels["square"] = np.sign(np.sin(2 * np.pi * t / 24.0)) * 0.8 + rng.normal(0, 0.05, rows)
        channels["spikes"] = rng.normal(0, 0.05, rows) + (rng.random(rows) < 0.01) * rng.normal(0, 2.0, rows)
        channels["period12"] = np.sin(2 * np.pi * t / 12.0) + 0.2 * np.sin(2 * np.pi * t / 24.0)
        logistic = np.zeros(rows)
        x = 0.4
        for i in range(rows):
            x = 3.8 * x * (1 - x)
            logistic[i] = x
        channels["chaotic"] = logistic - logistic.mean() + rng.normal(0, 0.02, rows)
    df = pd.DataFrame(channels)
    df.insert(0, "date", pd.date_range("2016-01-01", periods=rows, freq="h").astype(str))
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled synthetic dataset.")
    parser.add_argument("--out", default="csv_data/synthetic_hetero.csv")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--mode", default="heterogeneous", choices=["heterogeneous", "homogeneous", "balanced"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_series(args.rows, args.mode, args.seed)
    df.to_csv(out, index=False)
    print(f"[Synthetic] {args.mode}: {df.shape} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
