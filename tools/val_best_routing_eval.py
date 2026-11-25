"""Validation-based per-channel expert selection, evaluated on the test split.

For a trained run:
  1. score every expert on every channel on the validation split,
  2. pick the best expert per channel (no test information),
  3. force that routing and evaluate on the test split.

This isolates how much of the routing headroom is recoverable without a
learned gate.

    python tools/val_best_routing_eval.py --run-dir results/<run> --val-windows 512 --test-windows 2449
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.generic_csv import GenericCSVDataset
from training.core import build_dataset, get_model_instance
from training.baselines import make_repeated_seasonal_baseline_bcl
from utils.postprocess import apply_linear_drift_correction


def _json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_ds(cfg, mode, train_ds=None):
    kwargs = dict(
        seq_len=cfg["pred_len"], csv_path=cfg["csv_path"], mode=mode,
        context_len=cfg["context_len"], pred_len=cfg["pred_len"], channels_first=True,
        add_time_features=True, add_cyc_features=True, label_path=None,
        target_only_diffusion=True, split_borders=cfg.get("split_borders"),
        split_ratios=cfg.get("split_ratios"), scaler_type=cfg.get("scaler_type", "minmax"),
    )
    if train_ds is not None:
        kwargs["train_ds"] = train_ds
    return build_dataset(GenericCSVDataset, **kwargs)


def _apply_drift(pred_bcl, past_bcl, cfg, num_channels):
    """Match the evaluation pipeline: per-window level correction."""
    if not getattr(_apply_drift, "_printed", False):
        print(f"[debug] pred={pred_bcl.shape} past={past_bcl.shape} channels={num_channels}")
        _apply_drift._printed = True
    if not bool(cfg.get("drift_correction", True)):
        return pred_bcl
    if pred_bcl.shape[1] != num_channels:
        pred_bcl = pred_bcl.transpose(0, 2, 1)
    if past_bcl.shape[1] != num_channels:
        past_bcl = past_bcl.transpose(0, 2, 1)
    alpha = float(cfg.get("drift_alpha", 0.75))
    tail = int(cfg.get("drift_tail_window", 24))
    out = []
    for i in range(pred_bcl.shape[0]):
        p = pred_bcl[i].transpose(0, 1)
        h = past_bcl[i].transpose(0, 1)
        out.append(apply_linear_drift_correction(h, p, target_indices=list(range(p.shape[1])), tail_window=tail, alpha=alpha).T)
    return np.stack(out, axis=0)


@torch.no_grad()
def _expert_scores(model, ds, indices, target_idx, pred_len, seasonal_period, device, batch_size=64, cfg=None):
    names = list(model.expert_names)
    scores = {name: [] for name in names}
    for start in range(0, len(indices), batch_size):
        batch = [ds[i] for i in indices[start:start + batch_size]]
        past = torch.stack([b["past"] for b in batch]).to(device)
        fut = torch.stack([b["future"] for b in batch]).to(device)
        kf = torch.stack([b["known_future"] for b in batch]).to(device)
        base = make_repeated_seasonal_baseline_bcl(past, pred_len, seasonal_period)
        ctx = model._build_condition_context(past, known_future=kf)
        raw = model._align_tensor(past, model.num_channels, resize="none")
        true = fut.cpu().numpy().transpose(0, 2, 1)[:, :, target_idx]
        for i, name in enumerate(names):
            expert = model.experts[i]
            expert_past = raw if getattr(expert, "uses_full_context", False) else ctx
            out = expert(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device), expert_past)
            pred_full = (out + base).cpu().numpy()
            pred_full = _apply_drift(pred_full, past.cpu().numpy(), cfg, model.num_channels)
            pred = pred_full.transpose(0, 2, 1)[:, :, target_idx]
            scores[name].append(((pred - true) ** 2).mean(axis=1))
    return {name: np.concatenate(v, axis=0) for name, v in scores.items()}


@torch.no_grad()
def _routed_rmse(model, ds, indices, target_idx, pred_len, seasonal_period, device, batch_size=64, cfg=None):
    preds, trues = [], []
    for start in range(0, len(indices), batch_size):
        batch = [ds[i] for i in indices[start:start + batch_size]]
        past = torch.stack([b["past"] for b in batch]).to(device)
        fut = torch.stack([b["future"] for b in batch]).to(device)
        kf = torch.stack([b["known_future"] for b in batch]).to(device)
        cl = torch.stack([b.get("coarse_label", torch.zeros(ds.num_channels, dtype=torch.long)) for b in batch]).to(device)
        base = make_repeated_seasonal_baseline_bcl(past, pred_len, seasonal_period)
        out = model(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device),
                    past, coarse_label=cl, known_future=kf, return_weights=False)
        pred_full = (out + base).cpu().numpy()
        pred_full = _apply_drift(pred_full, past.cpu().numpy(), cfg, model.num_channels)
        pred = pred_full.transpose(0, 2, 1)[:, :, target_idx]
        preds.append(pred)
        trues.append(fut.cpu().numpy().transpose(0, 2, 1)[:, :, target_idx])
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    per_channel = ((pred - true) ** 2).mean(axis=(0, 1))
    return float(np.sqrt(((pred - true) ** 2).mean())), np.sqrt(per_channel)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validation-based channel routing evaluation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--test-windows", type=int, default=2449)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    cfg = _json(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    test_ds = _make_ds(cfg, "test", train_ds)
    target_idx = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    names = list(getattr(train_ds, "target_cols", [f"ch{i}" for i in target_idx]))
    pred_len = int(cfg["pred_len"])
    seasonal_period = int(cfg.get("seasonal_period", 24))

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=target_idx,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()

    val_indices = list(range(min(args.val_windows, len(val_ds))))
    scores = _expert_scores(model, val_ds, val_indices, target_idx, pred_len, seasonal_period, device, args.batch_size, cfg)
    expert_names = list(model.expert_names)
    choice, rows = [], []
    for t, name in enumerate(names):
        best = min(expert_names, key=lambda k: float(scores[k][:, t].mean()))
        choice.append(expert_names.index(best))
        rows.append({"channel": name, "val_best_expert": best,
                     "val_rmse": float(np.sqrt(scores[best][:, t].mean()))})
        print(f"{name:8s} val_best={best:28s} rmse={float(np.sqrt(scores[best][:, t].mean())):.4f}")

    gate_rmse, _ = _routed_rmse(model, test_ds, list(range(min(args.test_windows, len(test_ds)))),
                                target_idx, pred_len, seasonal_period, device, args.batch_size, cfg)
    model.set_fixed_channel_experts(choice)
    fixed_rmse, fixed_per_channel = _routed_rmse(model, test_ds, list(range(min(args.test_windows, len(test_ds)))),
                                                 target_idx, pred_len, seasonal_period, device, args.batch_size, cfg)
    print(f"test RMSE | gate={gate_rmse:.4f} | val_best_routing={fixed_rmse:.4f}")
    for name, value in zip(names, fixed_per_channel):
        print(f"  {name:8s} {value:.4f}")

    out = Path(args.out) if args.out else run / "val_best_routing.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
