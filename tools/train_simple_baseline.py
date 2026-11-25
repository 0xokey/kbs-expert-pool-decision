"""Simple, strong baselines evaluated on the exact same protocol as the MoE.

Models
------
- `DLinear`: series decomposition + per-horizon linear map (Zeng et al., 2023).
- `SeasonalNaive`: repeat the last seasonal cycle of the context.
- `Persistence`: repeat the last observed step.
- `HistoryMean`: repeat the context mean.

The runner reuses `GenericCSVDataset` (same splits, scaler, context/pred) and
reports metrics in the same space as the main experiments (`eval_space`), so
numbers are directly comparable with `evaluation/Final*/metrics/summary.json`.

Usage:
    python tools/train_simple_baseline.py --config configs/bench_etth1_h96.json --model DLinear --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.generic_csv import GenericCSVDataset
from utils.metrics import MetricsCalculator
from utils.common import resolve_context_pred_lengths
from training.baselines import make_baseline_prediction_np


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        pad = self.kernel_size // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        x = torch.cat([front, x, end], dim=-1)
        return self.avg(x)


class DLinear(nn.Module):
    """DLinear with shared linear map across channels."""

    def __init__(self, context_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.decompsition = MovingAverage(kernel_size)
        self.linear_seasonal = nn.Linear(self.context_len, self.pred_len)
        self.linear_trend = nn.Linear(self.context_len, self.pred_len)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        # past: [B, C, L] -> [B, C, pred] (TSLib DLinear: no instance normalisation)
        trend = self.decompsition(past)
        seasonal = past - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


def _build_dataset(config, mode, train_ds=None):
    context_len, pred_len = resolve_context_pred_lengths(config)
    kwargs = dict(
        seq_len=pred_len,
        context_len=context_len,
        pred_len=pred_len,
        csv_file_path=str(config["csv_path"]),
        mode=mode,
        channels_first=True,
        add_time_features=bool(config.get("add_time_features", False)),
        add_cyc_features=bool(config.get("add_cyc_features", False)),
        label_path=None,
        target_only_diffusion=bool(config.get("target_only_diffusion", False)),
        split_borders=config.get("split_borders"),
        split_ratios=config.get("split_ratios"),
        scaler_type=str(config.get("scaler_type", "minmax")),
    )
    if train_ds is not None:
        kwargs["scaler"] = train_ds.scaler
        kwargs["feature_cols"] = train_ds.feature_cols
        kwargs["target_cols"] = train_ds.target_cols
        kwargs["time_col"] = train_ds.time_col
    return GenericCSVDataset(**kwargs)


def _stack_batch(dataset, indices):
    items = [dataset[i] for i in indices]
    past = torch.stack([item["past"] for item in items], dim=0).float()
    future = torch.stack([item["future"] for item in items], dim=0).float()
    return past, future


@torch.no_grad()
def _predict(model, dataset, indices, batch_size=256):
    device = next(model.parameters()).device
    preds, trues = [], []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        past, future = _stack_batch(dataset, batch_idx)
        out = model(past.to(device))
        preds.append(out.detach().cpu().numpy())
        trues.append(future.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _metrics(pred, true, target_idx=None):
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim == 3:
        pred = pred.transpose(0, 2, 1)  # [N, C, L] -> [N, L, C]
        true = true.transpose(0, 2, 1)
    if target_idx is None:
        target_idx = list(range(pred.shape[-1]))
    p = pred[:, :, target_idx].reshape(-1)
    t = true[:, :, target_idx].reshape(-1)
    return {k: float(v) for k, v in MetricsCalculator.compute(t, p).items()}


def _per_channel(pred, true, target_names):
    rows = []
    for c, name in enumerate(target_names):
        p = pred[:, :, c].reshape(-1)
        t = true[:, :, c].reshape(-1)
        m = MetricsCalculator.compute(t, p)
        rows.append({"channel_idx": c, "channel_name": name, **{k: float(v) for k, v in m.items()}})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate simple forecasting baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="DLinear", choices=["DLinear", "SeasonalNaive", "Persistence", "HistoryMean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_len, pred_len = resolve_context_pred_lengths(config)
    train_ds = _build_dataset(config, "train")
    val_ds = _build_dataset(config, "val", train_ds)
    test_ds = _build_dataset(config, "test", train_ds)
    target_idx = getattr(train_ds, "target_indices", None) or list(range(train_ds.num_channels))
    target_names = list(getattr(train_ds, "target_cols", None) or [f"ch{i}" for i in target_idx])
    data_channels = int(train_ds.num_channels)

    base_tag = str(config.get("run_tag", Path(args.config).stem))
    tag = f"{base_tag}_{args.model}"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / f"Baseline_{args.model}_{Path(config['csv_path']).stem}_{tag}_s{args.seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Baseline] model={args.model} config={args.config} device={device}")
    print(f"[Baseline] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} channels={data_channels}")

    if args.model == "DLinear":
        model = DLinear(context_len, pred_len).to(device)
        train_indices = list(range(len(train_ds)))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            order = np.random.permutation(len(train_indices))
            epoch_loss = 0.0
            for start in range(0, len(order), args.batch_size):
                batch_idx = [train_indices[i] for i in order[start: start + args.batch_size]]
                past, future = _stack_batch(train_ds, batch_idx)
                past, future = past.to(device), future.to(device)
                pred = model(past)
                loss = loss_fn(pred, future)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            val_indices = list(range(len(val_ds)))
            val_pred, val_true = _predict(model, val_ds, val_indices)
            val_rmse = _metrics(val_pred, val_true, target_idx)["RMSE"]
            print(f"  epoch {epoch + 1}/{args.epochs} | train_mse={epoch_loss / max(1, math.ceil(len(order) / args.batch_size)):.5f} | val_rmse={val_rmse:.4f}")
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), run_dir / "model_weights.pth")
        val_pred, val_true = _predict(model, val_ds, list(range(len(val_ds))))
        test_pred, test_true = _predict(model, test_ds, list(range(len(test_ds))))
    else:
        method_map = {"SeasonalNaive": "daily_seasonal", "Persistence": "persistence", "HistoryMean": "history_mean"}
        method = method_map[args.model]
        seasonal_period = int(config.get("seasonal_period", 24))

        def _naive_predict(dataset):
            preds, trues = [], []
            for idx in range(len(dataset)):
                item = dataset[idx]
                past = item["past"].transpose(0, 1).numpy()  # [L, C]
                future = item["future"].transpose(0, 1).numpy()
                pred = make_baseline_prediction_np(method, past, pred_len=pred_len, seasonal_period=seasonal_period)
                preds.append(pred.T)
                trues.append(future.T)
            return np.stack(preds), np.stack(trues)

        val_pred, val_true = _naive_predict(val_ds)
        test_pred, test_true = _naive_predict(test_ds)

    val_metrics = _metrics(val_pred, val_true, target_idx)
    test_metrics = _metrics(test_pred, test_true, target_idx)
    payload = {
        "model": args.model,
        "run_tag": tag,
        "config": str(args.config),
        "seed": int(args.seed),
        "context_len": int(context_len),
        "pred_len": int(pred_len),
        "eval_space": str(config.get("eval_space", "raw")),
        "val": val_metrics,
        "test": test_metrics,
        "target_names": target_names,
    }
    with open(run_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    for split, rows in (("val", _per_channel(val_pred, val_true, target_names)), ("test", _per_channel(test_pred, test_true, target_names))):
        with open(run_dir / f"channel_metrics_{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[Baseline] val  RMSE={val_metrics['RMSE']:.4f} MAE={val_metrics['MAE']:.4f} R2={val_metrics['R2']:.4f}")
    print(f"[Baseline] test RMSE={test_metrics['RMSE']:.4f} MAE={test_metrics['MAE']:.4f} R2={test_metrics['R2']:.4f}")
    print(f"[Baseline] saved -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
