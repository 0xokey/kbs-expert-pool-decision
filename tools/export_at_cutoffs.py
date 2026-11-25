"""Export expert losses only at the forecast cutoffs used by the aligned pool."""
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
    p.add_argument("--cutoffs", required=True, help="npz with cut_val/cut_test")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = _make_ds(cfg, "train"); va = _make_ds(cfg, "val", tr); te = _make_ds(cfg, "test", tr)
    ti = list(getattr(tr, "target_indices", range(tr.num_channels)))
    pl = int(cfg["pred_len"]); per = int(cfg.get("seasonal_period", 24))
    z = np.load(args.cutoffs)
    want_val = set(z["cut_val"].tolist()); want_test = set(z["cut_test"].tolist())

    def indices(ds, want):
        out, keys = [], []
        for i in range(len(ds)):
            hist, _ = ds.get_time_window(i)
            h = hist[-1]
            key = np.datetime64(h, "ns").astype("int64") if isinstance(h, (np.datetime64, str)) else int(h)
            if key in want:
                out.append(i)
                keys.append(key)
        return out, np.array(keys, dtype=np.int64)

    iv, kv = indices(va, want_val); it, kt = indices(te, want_test)
    print(f"matched val={len(iv)}/{len(want_val)} test={len(it)}/{len(want_test)}")
    model = get_model_instance(cfg, tr.num_channels, device, target_indices=ti,
                               known_future_channels=tr.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    names = list(model.expert_names)
    val = _expert_channel_scores(model, va, iv, ti, pl, per, cfg, device, batch_size=args.batch_size)
    test = _expert_channel_scores(model, te, it, ti, pl, per, cfg, device, batch_size=args.batch_size)
    dest = ROOT / "results" / f"losses_{args.dataset}.npz"
    np.savez_compressed(dest,
                        val=np.stack([val[k] for k in names]),
                        test=np.stack([test[k] for k in names]),
                        names=np.array(names),
                        val_cut=kv, test_cut=kt)
    print(f"wrote {dest} val={len(iv)} test={len(it)} experts={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
