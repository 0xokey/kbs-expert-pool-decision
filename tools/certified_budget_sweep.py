"""Validation-budget sweep for certified vs naive routing.

For n in a grid, select per channel on the first n validation windows with
(a) naive val-best and (b) the paired certified rule (commit only when the top-2
experts are separated by more than the paired noise, else fall back to the
globally best expert).  Evaluate both on the full test split.

  python tools/certified_budget_sweep.py --run-dir results/<run> --dataset ETTh1
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

from tools.routing_analysis import _make_ds, _expert_channel_scores
from training.core import get_model_instance


def main(argv=None):
    parser = argparse.ArgumentParser(description="Certified vs naive routing budget sweep.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-mode", default="first", choices=["first", "last"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    test_ds = _make_ds(cfg, "test", train_ds)
    ti = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    pred_len = int(cfg["pred_len"])
    period = int(cfg.get("seasonal_period", 24))

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=ti,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    names = list(model.expert_names)
    C = len(ti)

    n_val_pool = min(max(args.sizes), len(val_ds))
    if args.window_mode == "last":
        val_idx = list(range(len(val_ds) - n_val_pool, len(val_ds)))
    else:
        val_idx = list(range(n_val_pool))
    val = _expert_channel_scores(model, val_ds, val_idx, ti, pred_len, period, cfg, device, batch_size=args.batch_size)
    test = _expert_channel_scores(model, test_ds, list(range(len(test_ds))), ti, pred_len, period, cfg, device, batch_size=args.batch_size)

    test_mse = np.stack([test[k].mean(axis=0) for k in names])  # [E, C]
    oracle = float(np.sqrt(test_mse.min(axis=0).mean()))

    rows = []
    for n in args.sizes:
        if n > n_val_pool:
            continue
        val_mse = np.stack([val[k][:n].mean(axis=0) for k in names])  # [E, C]
        naive_pick = np.argmin(val_mse, axis=0)
        naive = float(np.sqrt(test_mse[naive_pick, np.arange(C)].mean()))
        best_global = int(np.argmin(val_mse.mean(axis=1)))
        chosen = np.full(C, best_global, dtype=np.int64)
        certified = 0
        for c in range(C):
            order = np.argsort(val_mse[:, c])
            i1, i2 = int(order[0]), int(order[1])
            x1 = val[names[i1]][:n, c]
            x2 = val[names[i2]][:n, c]
            s1 = float(x1.std(ddof=1)) if n > 1 else 0.0
            s2 = float(x2.std(ddof=1)) if n > 1 else 0.0
            se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
            ref = max(abs(float(x1.mean())), 1e-8)
            margin = (float(x2.mean()) - float(x1.mean())) / ref
            threshold = args.z * se / ref
            if margin > threshold:
                chosen[c] = i1
                certified += 1
        cert = float(np.sqrt(test_mse[chosen, np.arange(C)].mean()))
        rows.append({"dataset": args.dataset, "n": n, "naive": naive, "certified": cert,
                     "oracle": oracle, "certified_channels": certified, "channels": C,
                     "gain_cert_vs_naive": (naive - cert) / naive * 100})
        print(f"{args.dataset} n={n:5d} naive={naive:.4f} certified={cert:.4f} "
              f"oracle={oracle:.4f} cert={certified}/{C} "
              f"gain={(naive - cert) / naive * 100:+.2f}%")

    out = Path(args.out) if args.out else run / "certified_budget_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("written ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
