"""Derive coarse regime labels from observed history windows.

The hard-channel gate consumes a 4-way coarse label
(0 stable, 1 volatile, 2 trend, 3 seasonal) per window and target channel.
This script derives those labels deterministically from the *past* window
only, so the same procedure applies to every dataset and split:

    vol    = std(first difference) / mean(|level|)
    trend  = |OLS slope| * L / mean(|level|)
    period = normalized autocorrelation at the seasonal lag

Per-channel thresholds are quantiles of the training split, so no validation
or test information leaks into the labels. Outputs follow the loader contract:

    <label-dir>/<dataset>_<mode>_labels.npy   int64 [num_windows, num_targets]
    <label-dir>/<dataset>_label_meta.json

Usage:
    python tools/make_coarse_labels.py --config configs/bench_etth1_h96.json
    python tools/make_coarse_labels.py --config ... --label-dir labels_derived
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.generic_csv import GenericCSVDataset


CLASS_NAMES = {0: "stable", 1: "volatile", 2: "trend", 3: "seasonal"}


def _window_stats(past: np.ndarray, period: int):
    """Return per-channel (volatility, trend, seasonality strength)."""
    past = np.asarray(past, dtype=np.float64)
    level = np.mean(np.abs(past), axis=0) + 1e-6
    vol = np.std(np.diff(past, axis=0), axis=0) / level

    t = np.arange(past.shape[0], dtype=np.float64)
    t_centered = t - t.mean()
    denom_t = float((t_centered ** 2).sum()) + 1e-8
    x_centered = past - past.mean(axis=0, keepdims=True)
    slope = (t_centered[:, None] * x_centered).sum(axis=0) / denom_t
    trend = np.abs(slope) * past.shape[0] / level

    lag = int(max(1, int(period)))
    if lag < past.shape[0]:
        a = past[:-lag, :]
        b = past[lag:, :]
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
        denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0)) + 1e-8
        period_strength = (a * b).sum(axis=0) / denom
    else:
        period_strength = np.zeros(past.shape[1], dtype=np.float64)
    return vol, trend, np.clip(period_strength, -1.0, 1.0)


def _collect_stats(dataset: GenericCSVDataset, period: int):
    data = dataset.data_tensor.detach().cpu().numpy()
    context_len = dataset.context_len
    vol_rows, trend_rows, period_rows = [], [], []
    for idx in range(len(dataset)):
        past = data[idx: idx + context_len]
        vol, trend, seasonality = _window_stats(past, period)
        vol_rows.append(vol)
        trend_rows.append(trend)
        period_rows.append(seasonality)
    return (
        np.stack(vol_rows) if vol_rows else np.zeros((0, data.shape[1])),
        np.stack(trend_rows) if trend_rows else np.zeros((0, data.shape[1])),
        np.stack(period_rows) if period_rows else np.zeros((0, data.shape[1])),
    )


def _label_windows(vol, trend, period, thresholds):
    labels = np.zeros_like(vol, dtype=np.int64)
    labels[vol >= thresholds["vol_hi"]] = 1
    labels[trend >= thresholds["trend_hi"]] = 2
    seasonal = (period >= thresholds["period_hi"]) & (period >= 0.3)
    labels[seasonal] = 3
    return labels


def _quantile(values: np.ndarray, q: float, fallback: float) -> np.ndarray:
    if values.size == 0:
        return np.full(values.shape[1] if values.ndim == 2 else 1, fallback, dtype=np.float64)
    return np.quantile(values, q, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive coarse labels from history windows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--vol-quantile", type=float, default=0.8)
    parser.add_argument("--trend-quantile", type=float, default=0.8)
    parser.add_argument("--period-quantile", type=float, default=0.6)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    csv_path = str(config.get("csv_path", ""))
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    base = Path(csv_path).stem
    label_dir = Path(args.label_dir or config.get("label_dir") or "labels_derived")
    label_dir.mkdir(parents=True, exist_ok=True)

    context_len = int(config.get("context_len", config.get("seq_len", 336)))
    pred_len = int(config.get("pred_len", 96))
    seasonal_period = int(config.get("seasonal_period", 24))
    split_borders = config.get("split_borders")
    split_ratios = config.get("split_ratios")
    scaler_type = str(config.get("scaler_type", "standard"))

    def build(mode, scaler=None):
        return GenericCSVDataset(
            seq_len=pred_len,
            context_len=context_len,
            pred_len=pred_len,
            csv_file_path=csv_path,
            mode=mode,
            channels_first=True,
            add_time_features=False,
            add_cyc_features=False,
            target_only_diffusion=True,
            scaler=scaler,
            split_borders=split_borders,
            split_ratios=split_ratios,
            scaler_type=scaler_type,
        )

    train_ds = build("train")
    vol_tr, trend_tr, period_tr = _collect_stats(train_ds, seasonal_period)
    thresholds = {
        "vol_hi": _quantile(vol_tr, args.vol_quantile, 0.0).tolist(),
        "trend_hi": _quantile(trend_tr, args.trend_quantile, 0.0).tolist(),
        "period_hi": _quantile(period_tr, args.period_quantile, 0.5).tolist(),
    }

    meta = {
        "csv_path": csv_path,
        "context_len": context_len,
        "pred_len": pred_len,
        "seasonal_period": seasonal_period,
        "scaler_type": scaler_type,
        "split_borders": list(train_ds.split_borders),
        "target_cols": list(train_ds.target_cols),
        "label_schema": CLASS_NAMES,
        "method": "history-window volatility / OLS trend / seasonal autocorrelation",
        "thresholds": thresholds,
        "splits": {},
    }

    for mode in ("train", "val", "test"):
        ds = build(mode, scaler=train_ds.scaler if mode != "train" else None)
        vol, trend, period = _collect_stats(ds, seasonal_period)
        labels = _label_windows(vol, trend, period, {k: np.asarray(v) for k, v in thresholds.items()})
        out_path = label_dir / f"{base}_{mode}_labels.npy"
        np.save(out_path, labels)
        counts = np.bincount(labels.reshape(-1), minlength=4).tolist()
        meta["splits"][mode] = {
            "file": str(out_path),
            "num_windows": int(labels.shape[0]),
            "num_targets": int(labels.shape[1]),
            "class_counts": counts,
        }
        print(f"[Labels] {mode}: {labels.shape} -> {out_path} counts={counts}")

    meta_path = label_dir / f"{base}_label_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Labels] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
