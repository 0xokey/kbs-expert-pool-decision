"""Export per-expert, per-channel, per-window losses for a trained pool."""
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
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-n", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = _make_ds(cfg, "train")
    va = _make_ds(cfg, "val", tr)
    te = _make_ds(cfg, "test", tr)
    ti = list(getattr(tr, "target_indices", range(tr.num_channels)))
    pl = int(cfg["pred_len"]); per = int(cfg.get("seasonal_period", 24))
    model = get_model_instance(cfg, tr.num_channels, device, target_indices=ti,
                               known_future_channels=tr.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    names = list(model.expert_names)
    n_val = min(args.val_n, len(va))
    val = _expert_channel_scores(model, va, list(range(n_val)), ti, pl, per, cfg, device,
                                 batch_size=args.batch_size)
    test = _expert_channel_scores(model, te, list(range(len(te))), ti, pl, per, cfg, device,
                                  batch_size=args.batch_size)
    def cutoffs(ds, n):
        out = []
        for i in range(n):
            hist, _ = ds.get_time_window(i)
            h = hist[-1]
            out.append(np.datetime64(h, "ns").astype("int64") if isinstance(h, (np.datetime64, str)) else int(h))
        return np.array(out, dtype=np.int64)
    out = {"dataset": args.dataset, "names": names,
           "val": np.stack([val[k] for k in names]),
           "test": np.stack([test[k] for k in names]),
           "val_cut": cutoffs(va, n_val), "test_cut": cutoffs(te, len(te))}
    dest = ROOT / "results" / f"losses_{args.dataset}.npz"
    np.savez_compressed(dest, dataset=args.dataset, names=np.array(names),
                        val=out["val"], test=out["test"],
                        val_cut=out["val_cut"], test_cut=out["test_cut"])
    print(f"{args.dataset}: {len(names)} experts, val {out['val'].shape}, test {out['test'].shape} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
