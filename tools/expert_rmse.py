"""Test RMSE of a single expert inside a run (same evaluation function as the paper)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.routing_analysis import _make_ds, _expert_channel_scores
from training.core import get_model_instance


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--expert", default="DLinearDiffuser")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    test_ds = _make_ds(cfg, "test", train_ds)
    ti = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=ti,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    scores = _expert_channel_scores(model, test_ds, list(range(len(test_ds))), ti,
                                    int(cfg["pred_len"]), int(cfg.get("seasonal_period", 24)),
                                    cfg, device, batch_size=args.batch_size)
    rmse = float(np.sqrt(scores[args.expert].mean()))
    print(f"{run.name} expert={args.expert} RMSE={rmse:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
