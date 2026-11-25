"""Per-expert, per-channel analysis for a routed run.

Answers two questions the paper needs:
  1. How good is every expert on every target channel (test split)?
  2. How much headroom does the gate leave?  We compare
       - the gate's routed prediction,
       - the best single expert per channel (oracle routing),
       - the globally best single expert.

    python tools/expert_oracle_analysis.py --run-dir results/<run> --max-windows 512
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
from utils.metrics import MetricsCalculator


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-expert per-channel analysis.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    cfg = _read_json(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(
        GenericCSVDataset, seq_len=cfg["pred_len"], csv_path=cfg["csv_path"], mode="train",
        context_len=cfg["context_len"], pred_len=cfg["pred_len"], channels_first=True,
        add_time_features=True, add_cyc_features=True, label_path=None,
        target_only_diffusion=True, split_borders=cfg.get("split_borders"),
        split_ratios=cfg.get("split_ratios"), scaler_type=cfg.get("scaler_type", "minmax"),
    )
    ds = build_dataset(
        GenericCSVDataset, seq_len=cfg["pred_len"], csv_path=cfg["csv_path"], mode=args.split,
        train_ds=train_ds, context_len=cfg["context_len"], pred_len=cfg["pred_len"],
        channels_first=True, add_time_features=True, add_cyc_features=True, label_path=None,
        target_only_diffusion=True, split_borders=cfg.get("split_borders"),
        split_ratios=cfg.get("split_ratios"), scaler_type=cfg.get("scaler_type", "minmax"),
    )
    model = get_model_instance(
        cfg, train_ds.num_channels, device, target_indices=train_ds.target_indices,
        known_future_channels=train_ds.known_future_channels,
    )
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()

    n = min(int(args.max_windows), len(ds))
    indices = list(range(n))
    target_idx = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    names = list(getattr(train_ds, "target_cols", [f"ch{i}" for i in target_idx]))
    pred_len = int(cfg["pred_len"])

    expert_names = list(model.expert_names)
    per_expert = {name: np.zeros((0, len(target_idx))) for name in expert_names}
    routed = []
    truth = []

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch = [ds[i] for i in indices[start:start + args.batch_size]]
            past = torch.stack([b["past"] for b in batch]).to(device)
            fut = torch.stack([b["future"] for b in batch]).to(device)
            kf = torch.stack([b["known_future"] for b in batch]).to(device)
            cl = torch.stack([b.get("coarse_label", torch.zeros(ds.num_channels, dtype=torch.long)) for b in batch]).to(device)
            base = make_repeated_seasonal_baseline_bcl(past, pred_len, cfg.get("seasonal_period", 24))
            ctx = model._build_condition_context(past, known_future=kf)
            raw = model._align_tensor(past, model.num_channels, resize="none")
            for i, name in enumerate(expert_names):
                expert = model.experts[i]
                expert_past = raw if getattr(expert, "uses_full_context", False) else ctx
                out = expert(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device), expert_past)
                pred = (out + base).detach().cpu().numpy()       # [B, C, L]
                true = fut.detach().cpu().numpy()
                pred = pred.transpose(0, 2, 1)[:, :, target_idx]  # [B, L, T]
                true = true.transpose(0, 2, 1)[:, :, target_idx]
                err = ((pred - true) ** 2).mean(axis=1)           # [B, T] per-channel MSE
                per_expert[name] = np.concatenate([per_expert[name], err], axis=0)
            with torch.no_grad():
                routed_out = model(
                    torch.zeros_like(fut),
                    torch.zeros(fut.size(0), dtype=torch.long, device=device),
                    past, coarse_label=cl, known_future=kf, return_weights=False,
                )
            routed_pred = (routed_out + base).detach().cpu().numpy().transpose(0, 2, 1)[:, :, target_idx]
            routed.append(((routed_pred - true) ** 2).mean(axis=1))
            truth.append(true)

    routed_mse = np.concatenate(routed, axis=0).mean(axis=0)     # [T]
    oracle_mse = np.full(len(target_idx), np.inf)
    best_expert = [""] * len(target_idx)
    rows = []
    for t, name in enumerate(names):
        scores = {expert: float(per_expert[expert][:, t].mean()) for expert in expert_names}
        best = min(scores, key=scores.get)
        oracle_mse[t] = scores[best]
        best_expert[t] = best
        rows.append({
            "channel": name,
            "gate_rmse": float(np.sqrt(routed_mse[t])),
            "oracle_rmse": float(np.sqrt(scores[best])),
            "oracle_expert": best,
            "global_best_expert": min(
                {expert: float(per_expert[expert].mean()) for expert in expert_names}, key=lambda k: per_expert[k].mean()
            ),
        })
        print(f"{name:8s} gate={np.sqrt(routed_mse[t]):.4f} oracle={np.sqrt(scores[best]):.4f} ({best})")

    gate_overall = float(np.sqrt(routed_mse.mean()))
    oracle_overall = float(np.sqrt(oracle_mse.mean()))
    print(f"overall: gate={gate_overall:.4f} oracle={oracle_overall:.4f}")

    out_path = Path(args.out) if args.out else run / f"expert_oracle_{args.split}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "gate_rmse", "oracle_rmse", "oracle_expert", "global_best_expert"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"written -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
