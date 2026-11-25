"""Active Certified Channel Routing (Active-CCR).

Instead of spending a fixed validation budget on every channel, Active-CCR
certifies channels sequentially: it starts from a small pool of validation
windows, and for every channel whose top-two experts are not yet separated it
draws additional windows, until the two-expert confidence intervals are
disjoint or the per-channel budget is exhausted.  Channels that are certified are
committed to their best expert; the rest fall back to a validated safe predictor.

The tool simulates the active process over precomputed per-window losses, so it
only needs an existing checkpoint.

    python tools/active_certified_routing.py --run-dir results/<run> --n0 32 --step 32 --max-n 512
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

from tools.routing_analysis import _make_ds, _expert_channel_scores, _routed_channel_scores, _rmse_from_scores
from training.core import get_model_instance


def _json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _certify_channel(scores, names, c, n0, step, max_n, z, rng):
    """Sequentially sample validation windows until the top-2 are separated."""
    pool = np.arange(scores[names[0]].shape[0])
    used = rng.choice(pool, size=min(n0, len(pool)), replace=False)
    n = len(used)
    while True:
        vec = np.array([float(scores[k][used, c].mean()) for k in names])
        order = np.argsort(vec)
        best, second = order[0], order[1]
        r1, r2 = float(vec[best]), float(vec[second])
        s1 = float(scores[names[best]][used, c].std(ddof=1)) if n > 1 else 0.0
        s2 = float(scores[names[second]][used, c].std(ddof=1)) if n > 1 else 0.0
        se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
        margin = (r2 - r1) / max(r1, 1e-8)
        threshold = z * se / max(r1, 1e-8)
        certified = margin > threshold
        if certified or n >= max_n or n >= len(pool):
            return int(best), bool(certified), n, margin, threshold
        remaining = np.setdiff1d(pool, used, assume_unique=False)
        if len(remaining) == 0:
            return int(best), True, n, margin, threshold
        add = rng.choice(remaining, size=min(step, len(remaining)), replace=False)
        used = np.union1d(used, add)
        n = len(used)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Active certified channel routing.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n0", type=int, default=32)
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--max-n", type=int, default=512)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--fallback", default="best_single", choices=["topk", "uniform", "best_single"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
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
    period = int(cfg.get("seasonal_period", 24))

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=target_idx,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    expert_names = list(model.expert_names)
    test_indices = list(range(len(test_ds)))

    val_scores = _expert_channel_scores(model, val_ds, list(range(len(val_ds))), target_idx, pred_len, period, cfg, device, batch_size=args.batch_size)
    test_scores = _expert_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size)

    model.clear_fixed_channel_experts()
    gate_rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size))
    singles = {}
    for i, name in enumerate(expert_names):
        model.set_fixed_channel_experts([i] * len(target_idx))
        singles[name] = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size))
    best_single = min(singles.values())
    # Deployable single-expert baseline: chosen on the validation split.
    val_mean = np.array([float(val_scores[k].mean()) for k in expert_names])
    val_best_idx = int(np.argmin(val_mean))
    model.set_fixed_channel_experts([val_best_idx] * len(target_idx))
    val_best_single = _rmse_from_scores(
        _routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size)
    )
    oracle_choice = [int(np.argmin([float(test_scores[k][:, c].mean()) for k in expert_names])) for c in range(len(names))]
    model.set_fixed_channel_experts(oracle_choice)
    oracle_rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size))

    print(f"gate={gate_rmse:.4f} best_single_test={best_single:.4f} "
          f"best_single_val={val_best_single:.4f} oracle={oracle_rmse:.4f}")
    rows = []
    for rep in range(args.repeats):
        rng = np.random.default_rng(rep)
        W = np.zeros((len(names), len(expert_names)), dtype=np.float32)
        certified = 0
        used_ns = []
        for c in range(len(names)):
            best, cert, n, margin, threshold = _certify_channel(
                val_scores, expert_names, c, args.n0, args.step, args.max_n, args.z, rng
            )
            used_ns.append(n)
            if cert:
                certified += 1
                W[c, best] = 1.0
            else:
                if args.fallback == "topk":
                    vec = np.array([float(val_scores[k][:, c].mean()) for k in expert_names])
                    pick = np.argsort(vec)[: max(1, args.topk)]
                    W[c, pick] = 1.0 / len(pick)
                elif args.fallback == "uniform":
                    W[c, :] = 1.0 / len(expert_names)
                else:
                    mean_vec = np.array([float(val_scores[k].mean()) for k in expert_names])
                    W[c, int(np.argmin(mean_vec))] = 1.0
        model.set_fixed_channel_weights(W)
        rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device, batch_size=args.batch_size))
        gain = (val_best_single - rmse) / val_best_single
        rows.append({"repeat": rep, "rmse": rmse, "gain": gain,
                     "best_single_val": val_best_single, "best_single_test": best_single,
                     "gate": gate_rmse, "oracle": oracle_rmse, "certified": certified,
                     "channels": len(names), "mean_n": float(np.mean(used_ns)),
                     "max_n": int(np.max(used_ns))})
        print(f"rep={rep} active_ccr={rmse:.4f} gain={gain:+.4f} certified={certified}/{len(names)} "
              f"mean_n={np.mean(used_ns):.0f} max_n={np.max(used_ns)}")

    out = Path(args.out) if args.out else run / f"active_ccr_{args.fallback}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
