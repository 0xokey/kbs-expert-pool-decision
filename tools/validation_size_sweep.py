"""Validation-size sweep: how many validation windows does routing need?

For a trained run we select, per channel, the expert with the lowest validation
error computed on the first n windows, freeze that assignment, and measure the
test RMSE. Repeating for a grid of n gives the empirical selection curve that
the theory in the paper predicts: selection needs n ~ (sigma/delta)^2 to recover
the oracle.

    python tools/validation_size_sweep.py --run-dir results/<run> --sizes 16 32 64 128 256 512
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

from tools.routing_analysis import _make_ds, _drift, _routed_channel_scores, _rmse_from_scores, _expert_channel_scores
from training.core import get_model_instance


def _json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validation size sweep for routing.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=3)
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
    period = int(cfg.get("seasonal_period", 24))

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=target_idx,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    expert_names = list(model.expert_names)

    test_indices = list(range(len(test_ds)))
    test_scores = _expert_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device)
    model.clear_fixed_channel_experts()
    gate_rmse = _rmse_from_scores(
        _routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device)
    )
    singles = {}
    for i, name in enumerate(expert_names):
        model.set_fixed_channel_experts([i] * len(target_idx))
        singles[name] = _rmse_from_scores(
            _routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device)
        )
    best_single = min(singles.values())

    rows = []
    rng = np.random.default_rng(0)
    for n in args.sizes:
        if n > len(val_ds):
            continue
        gains, accs = [], []
        for rep in range(args.repeats):
            if n == len(val_ds):
                idx = list(range(n))
            else:
                idx = sorted(rng.choice(len(val_ds), size=n, replace=False).tolist())
            scores = _expert_channel_scores(model, val_ds, idx, target_idx, pred_len, period, cfg, device)
            choice, acc = [], 0
            for t in range(len(names)):
                vec = np.array([float(scores[k][:, t].mean()) for k in expert_names])
                pick = int(np.argmin(vec))
                choice.append(pick)
                test_vec = np.array([float(test_scores[k][:, t].mean()) for k in expert_names])
                acc += int(pick == int(np.argmin(test_vec)))
            model.set_fixed_channel_experts(choice)
            rmse = _rmse_from_scores(
                _routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device)
            )
            gains.append((best_single - rmse) / best_single)
            accs.append(acc / max(len(names), 1))
        rows.append({
            "run": run.name,
            "n_val": int(n),
            "gain_mean": float(np.mean(gains)),
            "gain_std": float(np.std(gains)),
            "selection_accuracy": float(np.mean(accs)),
            "best_single": float(best_single),
            "gate": float(gate_rmse),
        })
        print(f"n={n:4d} gain={np.mean(gains):+.4f}±{np.std(gains):.4f} "
              f"acc={np.mean(accs):.2f} (best_single={best_single:.4f}, gate={gate_rmse:.4f})")

    out = Path(args.out) if args.out else run / "validation_size_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
