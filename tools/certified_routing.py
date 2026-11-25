"""Certified Channel Routing (CCR).

Post-hoc routing rule that uses the validation margin to decide, per channel,
whether to commit to one expert or to fall back to a mixture:

  1. estimate every expert on n validation windows;
  2. compute the margin m_c = (R_2 - R_1)/R_1 between the best and second best;
  3. if m_c >= tau, route the channel to the best expert (one-hot);
     otherwise fall back to a mixture (top-k by validation, uniform, or the
     globally best single expert).

The rule is motivated by the selection bound in the paper: a channel can only be
routed reliably when the margin exceeds the estimation error.  No retraining is
required; the tool only re-uses an existing checkpoint.

    python tools/certified_routing.py --run-dir results/<run> --val-windows 256
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


def _auto_thresholds(scores, names, z, n):
    """Per-channel relative margin threshold from a two-expert confidence test."""
    K = len(names)
    C = scores[names[0]].shape[1]
    taus = np.zeros(C, dtype=np.float32)
    for c in range(C):
        vec = np.array([float(scores[k][:, c].mean()) for k in names])
        order = np.argsort(vec)
        best, second = float(vec[order[0]]), float(vec[order[1]])
        s1 = float(scores[names[order[0]]][:, c].std(ddof=1))
        s2 = float(scores[names[order[1]]][:, c].std(ddof=1))
        se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
        taus[c] = float(z * se / max(best, 1e-8))
    return taus


def _weights(scores, names, target_names, tau, fallback, topk, per_channel_tau=None):
    K = len(names)
    C = len(target_names)
    W = np.zeros((C, K), dtype=np.float32)
    margins = []
    for c in range(C):
        vec = np.array([float(scores[k][:, c].mean()) for k in names])
        order = np.argsort(vec)
        best, second = float(vec[order[0]]), float(vec[order[1]])
        margin = (second - best) / max(best, 1e-8)
        margins.append(margin)
        threshold = tau if per_channel_tau is None else float(per_channel_tau[c])
        if margin >= threshold:
            W[c, int(order[0])] = 1.0
        else:
            if fallback == "topk":
                pick = order[: max(1, min(topk, K))]
                W[c, pick] = 1.0 / len(pick)
            elif fallback == "uniform":
                W[c, :] = 1.0 / K
            elif fallback == "best_single":
                mean_vec = np.array([float(scores[k].mean()) for k in names])
                W[c, int(np.argmin(mean_vec))] = 1.0
            else:
                raise ValueError(f"unknown fallback {fallback!r}")
    return W, margins


def main(argv=None):
    parser = argparse.ArgumentParser(description="Certified channel routing.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-windows", type=int, default=256)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--fallback", default="topk", choices=["topk", "uniform", "best_single"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tau-mode", default="manual", choices=["manual", "auto"])
    parser.add_argument("--z", type=float, default=1.96)
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
    val_indices = list(range(min(args.val_windows, len(val_ds))))

    val_scores = _expert_channel_scores(model, val_ds, val_indices, target_idx, pred_len, period, cfg, device)
    test_scores = _expert_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device)

    model.clear_fixed_channel_experts()
    gate_rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device))
    singles = {}
    for i, name in enumerate(expert_names):
        model.set_fixed_channel_experts([i] * len(target_idx))
        singles[name] = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device))
    best_single = min(singles.values())

    oracle_choice = []
    for c in range(len(names)):
        vec = np.array([float(test_scores[k][:, c].mean()) for k in expert_names])
        oracle_choice.append(int(np.argmin(vec)))
    model.set_fixed_channel_experts(oracle_choice)
    oracle_rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device))

    rows = [{"method": "gate", "rmse": gate_rmse, "tau": "", "fallback": ""},
            {"method": "best_single", "rmse": best_single, "tau": "", "fallback": ""},
            {"method": "oracle", "rmse": oracle_rmse, "tau": "", "fallback": ""}]
    print(f"gate={gate_rmse:.4f} best_single={best_single:.4f} oracle={oracle_rmse:.4f}")
    auto_taus = None
    if args.tau_mode == "auto":
        auto_taus = _auto_thresholds(val_scores, expert_names, args.z, len(val_indices))
        print(f"auto thresholds (z={args.z:.2f}): " + ", ".join(f"{v:.3f}" for v in auto_taus))
    taus = args.taus if args.tau_mode == "manual" else list(args.taus)
    for tau in taus:
        W, margins = _weights(
            val_scores, expert_names, names, tau, args.fallback, args.topk,
            per_channel_tau=auto_taus if args.tau_mode == "auto" else None,
        )
        model.set_fixed_channel_weights(W)
        rmse = _rmse_from_scores(_routed_channel_scores(model, test_ds, test_indices, target_idx, pred_len, period, cfg, device))
        committed = int((np.max(W, axis=1) > 0.999).sum())
        rows.append({"method": "ccr", "rmse": rmse, "tau": tau, "fallback": args.fallback,
                     "committed_channels": committed, "median_margin": float(np.median(margins))})
        print(f"tau={tau:.3f} ccr={rmse:.4f} (committed {committed}/{len(names)}, "
              f"median margin {np.median(margins):.3f}, gain {(best_single - rmse) / best_single:+.3f})")

    out = Path(args.out) if args.out else run / f"certified_routing_{args.fallback}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rmse", "tau", "fallback", "committed_channels", "median_margin"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
