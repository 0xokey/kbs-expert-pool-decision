"""Certified pruning analysis.

For a trained MoE run, apply the certified rule at a fixed validation budget and
report which experts are actually needed (distinct selections).  Experts that are
never selected can be removed at inference without changing a single prediction.

  python tools/prune_analysis.py --run-dir results/<run> --n 256
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Certified pruning analysis.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    ti = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    pred_len = int(cfg["pred_len"])
    period = int(cfg.get("seasonal_period", 24))
    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=ti,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    names = list(model.expert_names)
    C = len(ti)
    n = min(args.n, len(val_ds))
    val = _expert_channel_scores(model, val_ds, list(range(n)), ti, pred_len, period, cfg, device,
                                 batch_size=args.batch_size)
    val_mse = np.stack([val[k].mean(axis=0) for k in names])
    best_global = int(np.argmin(val_mse.mean(axis=1)))
    chosen = []
    for c in range(C):
        order = np.argsort(val_mse[:, c])
        i1, i2 = int(order[0]), int(order[1])
        d = val[names[i1]][:, c] - val[names[i2]][:, c]
        sd = float(d.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(max(n, 2))
        chosen.append(i1 if float(d.mean()) < -args.z * se else best_global)
    cnt = Counter(chosen)
    used = sorted(cnt)
    print(f"{run.name}: channels={C} n={n} distinct_experts={len(used)}/{len(names)}")
    for e, k in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"  {names[e]:28s} {k} channels")
    print("experts used:", ", ".join(names[e] for e in used))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
