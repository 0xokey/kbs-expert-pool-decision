"""MAE-based routing over an expert pool.

The diffusion experts here are deployed deterministically, so CRPS degenerates to
MAE; routing by an absolute-error criterion is therefore the meaningful
alternative to the MSE criterion.  Select per channel by validation MAE and
report test MAE and RMSE, with the corresponding oracles.

  python tools/metric_routing.py --run-dirs A B ... --val-windows 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.mix_routing import predict_expert
from tools.routing_analysis import _make_ds


def main(argv=None):
    parser = argparse.ArgumentParser(description="MAE-based routing.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--val-windows", type=int, default=1024)
    args = parser.parse_args(argv)

    names, Pv, Pt = [], [], []
    for d in args.run_dirs:
        name, v, t, _ = predict_expert(Path(d), args.val_windows)
        names.append(name)
        Pv.append(v)
        Pt.append(t)
        print("loaded", name)

    V = np.stack(Pv)
    T = np.stack(Pt)
    E, nv, C, H = V.shape

    cfg = json.load(open(Path(args.run_dirs[0]) / "config.json"))
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    test_ds = _make_ds(cfg, "test", train_ds)
    n_fit = min(args.val_windows, len(val_ds))
    Yv = np.stack([val_ds[i]["future"].numpy() for i in range(n_fit)])
    Y = np.stack([test_ds[i]["future"].numpy() for i in range(len(test_ds))])

    def mae(p, y):
        return float(np.abs(p - y).mean())

    def rmse(p, y):
        return float(np.sqrt(((p - y) ** 2).mean()))

    val_mae = np.array([[mae(V[e][:, c, :], Yv[:, c, :]) for c in range(C)] for e in range(E)])
    test_mae = np.array([[mae(T[e][:, c, :], Y[:, c, :]) for c in range(C)] for e in range(E)])
    test_rmse = np.array([[rmse(T[e][:, c, :], Y[:, c, :]) for c in range(C)] for e in range(E)])

    best = int(np.argmin(val_mae.mean(axis=1)))
    print("\nbest single (%s): MAE %.4f RMSE %.4f" % (names[best], test_mae[best].mean(), np.sqrt((test_rmse[best] ** 2).mean())))

    pick = np.argmin(val_mae, axis=0)
    routed_mae = float(test_mae[pick, np.arange(C)].mean())
    print("MAE-selected routing: MAE %.4f (%+.2f%% vs best single)" % (
        routed_mae, (test_mae[best].mean() - routed_mae) / test_mae[best].mean() * 100))

    oracle_mae = float(test_mae.min(axis=0).mean())
    print("MAE oracle: MAE %.4f (%+.2f%% vs best single)" % (
        oracle_mae, (test_mae[best].mean() - oracle_mae) / test_mae[best].mean() * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
