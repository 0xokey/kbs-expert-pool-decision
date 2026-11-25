"""Per-window paired test between two runs (same architecture, different training)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.routing_analysis import _make_ds, _expert_channel_scores
from training.core import get_model_instance


def window_losses(run_dir, expert, device, batch_size=64):
    run = Path(run_dir)
    cfg = json.load(open(run / "config.json"))
    train_ds = _make_ds(cfg, "train")
    test_ds = _make_ds(cfg, "test", train_ds)
    ti = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=ti,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    scores = _expert_channel_scores(model, test_ds, list(range(len(test_ds))), ti,
                                    int(cfg["pred_len"]), int(cfg.get("seasonal_period", 24)),
                                    cfg, device, batch_size=batch_size)
    return scores[expert].mean(axis=1)  # per-window loss


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="run A (e.g. isolated)")
    p.add_argument("--b", required=True, help="run B (e.g. in mixture)")
    p.add_argument("--expert", default="DLinearDiffuser")
    p.add_argument("--label", default="")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = window_losses(args.a, args.expert, device, args.batch_size)
    b = window_losses(args.b, args.expert, device, args.batch_size)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    w = stats.wilcoxon(a, b, alternative="less")
    t = stats.ttest_rel(a, b)
    print(f"{args.label} windows={n} mean_A={a.mean():.4f} mean_B={b.mean():.4f} "
          f"wilcoxon_p={w.pvalue:.3e} t_p={t.pvalue:.3e} rel_gain={100*(b.mean()-a.mean())/b.mean():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
