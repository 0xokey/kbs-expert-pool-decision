"""Train a strong baseline with neuralforecast and export per-channel losses.

The splits are the project's chronological borders; each channel is treated as an
independent series and the model forecasts the scaled future.  The script writes
per-channel MSE on the validation and test blocks, so the certified routing can
treat this model as one more expert in a heterogeneous pool.

  python tools/nf_bench.py --dataset ETTh1 --model nhits --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(dataset):
    hits = list((ROOT / "configs").glob(f"bench_{dataset.lower()}_h96_direct_es.json"))
    if not hits:
        raise SystemExit(f"no config for {dataset}")
    return json.load(open(hits[0]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="nhits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--input-size", type=int, default=336)
    args = p.parse_args(argv)
    cfg = load_config(args.dataset)
    df = pd.read_csv(ROOT / cfg["csv_path"])
    time_col = "date" if "date" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    borders = cfg.get("split_borders")
    tr_end, va_end, te_end = borders
    target_cols = [c for c in df.columns if c != time_col]

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, NBEATS, TSMixer, PatchTST, iTransformer

    models = {
        "nhits": lambda h, i: NHITS(h=h, input_size=i, max_steps=1000, batch_size=64,
                                    learning_rate=1e-3, random_seed=args.seed, accelerator="gpu"),
        "nbeats": lambda h, i: NBEATS(h=h, input_size=i, max_steps=1000, batch_size=64,
                                      learning_rate=1e-3, random_seed=args.seed, accelerator="gpu"),
        "tsmixer": lambda h, i: TSMixer(h=h, input_size=i, max_steps=1000, batch_size=64,
                                        learning_rate=1e-3, random_seed=args.seed, accelerator="gpu"),
        "patchtst": lambda h, i: PatchTST(h=h, input_size=i, max_steps=1000, batch_size=64,
                                          learning_rate=1e-4, random_seed=args.seed, accelerator="gpu"),
        "itransformer": lambda h, i: iTransformer(h=h, input_size=i, max_steps=1000, batch_size=64,
                                                  learning_rate=1e-4, random_seed=args.seed, accelerator="gpu"),
    }

    rows = []
    for c in target_cols:
        y = df[c].to_numpy(dtype=np.float64)
        mu, sd = y[:tr_end].mean(), y[:tr_end].std() + 1e-8
        ds = pd.DataFrame({"unique_id": c, "ds": df[time_col], "y": (y - mu) / sd})
        nf = NeuralForecast(models=[models[args.model](args.horizon, args.input_size)], freq="h")
        train = ds.iloc[:tr_end]
        nf.fit(train)
        val_pred = nf.predict()
        test_pred = nf.predict()
        fcst = [n for n in val_pred.columns if n not in ("unique_id", "ds")][0]

        def block_mse(start, end, pred_df):
            # predict() returns the next horizon after the fitted series; use the
            # score instead for a block-level loss when borders do not align.
            return None

        # block-level losses via cross-validation on the full series
        cv = nf.cross_validation(ds, n_windows=8, step_size=args.horizon, val_size=args.horizon * 2)
        val_windows = cv[cv["ds"] < df[time_col].iloc[va_end]]
        test_windows = cv[cv["ds"] >= df[time_col].iloc[va_end]]
        val_mse = float(((val_windows["y"] - val_windows[fcst]) ** 2).mean()) if len(val_windows) else float("nan")
        test_mse = float(((test_windows["y"] - test_windows[fcst]) ** 2).mean()) if len(test_windows) else float("nan")
        rows.append({"channel": c, "val_mse": val_mse, "test_mse": test_mse})

    out = {"dataset": args.dataset, "model": args.model, "seed": args.seed,
           "channels": [r["channel"] for r in rows],
           "val_mse": [r["val_mse"] for r in rows],
           "test_mse": [r["test_mse"] for r in rows]}
    dest = ROOT / "results" / f"nf_{args.model}_{args.dataset}_s{args.seed}.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps({"dataset": args.dataset, "model": args.model,
                      "val_rmse": float(np.sqrt(np.nanmean(out["val_mse"]))),
                      "test_rmse": float(np.sqrt(np.nanmean(out["test_mse"])))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
