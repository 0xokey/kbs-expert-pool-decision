"""Selection baselines under a fixed validation budget.

Compares, for each channel:
  naive      : argmin of the mean validation loss
  certified  : two-sample two-expert test (commit or fall back to the global best)
  cv         : 5-fold cross-validation, majority vote over fold winners
  bootstrap  : 200 bootstrap resamples, highest win rate
  naive2x    : naive selection with twice the budget (budget control)

Evaluation is the test RMSE of the resulting per-channel assignment.

  python tools/selection_baselines.py --run-dir results/<run> --dataset ETTh1
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
    parser = argparse.ArgumentParser(description="Selection baselines under a fixed budget.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--boot", type=int, default=200)
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

    n_pool = min(2 * max(args.sizes), len(val_ds))
    val = _expert_channel_scores(model, val_ds, list(range(n_pool)), ti, pred_len, period, cfg, device,
                                 batch_size=args.batch_size)
    test = _expert_channel_scores(model, test_ds, list(range(len(test_ds))), ti, pred_len, period, cfg, device,
                                  batch_size=args.batch_size)
    test_mse = np.stack([test[k].mean(axis=0) for k in names])
    oracle = float(np.sqrt(test_mse.min(axis=0).mean()))

    def rmse_for(pick):
        return float(np.sqrt(test_mse[pick, np.arange(C)].mean()))

    rng = np.random.default_rng(0)
    rows = []
    for n in args.sizes:
        if n > n_pool:
            continue
        V = np.stack([val[k][:n] for k in names])  # [E, n, C]
        val_mse = V.mean(axis=1)
        best_global = int(np.argmin(val_mse.mean(axis=1)))

        naive = rmse_for(np.argmin(val_mse, axis=0))

        cert = np.full(C, best_global, dtype=np.int64)
        for c in range(C):
            order = np.argsort(val_mse[:, c])
            i1, i2 = int(order[0]), int(order[1])
            x1 = V[i1, :, c]
            x2 = V[i2, :, c]
            s1 = float(x1.std(ddof=1)) if n > 1 else 0.0
            s2 = float(x2.std(ddof=1)) if n > 1 else 0.0
            se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
            ref = max(abs(float(x1.mean())), 1e-8)
            margin = (float(x2.mean()) - float(x1.mean())) / ref
            threshold = args.z * se / ref
            if margin > threshold:
                cert[c] = i1
        cert_r = rmse_for(cert)

        # 5-fold CV majority vote
        folds = np.array_split(np.arange(n), args.folds)
        votes = np.zeros((C, len(names)), dtype=int)
        for f in folds:
            fm = np.stack([V[e][f].mean(axis=0) for e in range(len(names))])
            for c in range(C):
                votes[c, int(np.argmin(fm[:, c]))] += 1
        cv_pick = np.argmax(votes, axis=1)
        cv_r = rmse_for(cv_pick)

        # bootstrap win rate
        wins = np.zeros((C, len(names)))
        for b in range(args.boot):
            idx = rng.integers(0, n, size=n)
            bm = np.stack([V[e][idx].mean(axis=0) for e in range(len(names))])
            for c in range(C):
                wins[c, int(np.argmin(bm[:, c]))] += 1
        boot_pick = np.argmax(wins, axis=1)
        boot_r = rmse_for(boot_pick)

        # budget control: naive with 2n windows
        if 2 * n <= n_pool:
            V2 = np.stack([val[k][:2 * n] for k in names])
            naive2x = rmse_for(np.argmin(V2.mean(axis=1), axis=0))
        else:
            naive2x = float("nan")

        rows.append({"dataset": args.dataset, "n": n, "naive": naive, "certified": cert_r,
                     "cv": cv_r, "bootstrap": boot_r, "naive2x": naive2x, "oracle": oracle})
        print(f"{args.dataset} n={n:4d} naive={naive:.4f} cert={cert_r:.4f} cv={cv_r:.4f} "
              f"boot={boot_r:.4f} naive2x={naive2x:.4f} oracle={oracle:.4f}")

    out = Path(args.out) if args.out else run / "selection_baselines.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("written ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
