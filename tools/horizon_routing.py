"""Per-horizon-segment routing over an expert pool.

For each channel and each horizon segment (e.g. 4 x 24 steps) the expert with the
lowest validation error on that segment is applied to the test split.  Requires
only forward passes.

  python tools/horizon_routing.py --run-dirs A B ... --segment 24 --val-windows 1024
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
    parser = argparse.ArgumentParser(description="Per-horizon-segment routing.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--val-windows", type=int, default=1024)
    parser.add_argument("--segment", type=int, default=24)
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
    nv = min(args.val_windows, len(val_ds))
    Yv = np.stack([val_ds[i]["future"].numpy() for i in range(nv)])          # [nv,C,H]
    Y = np.stack([test_ds[i]["future"].numpy() for i in range(len(test_ds))])  # [nt,C,H]

    def rmse(p):
        return float(np.sqrt(((p - Y) ** 2).mean()))

    singles = [rmse(T[e]) for e in range(E)]
    best = int(np.argmin(singles))
    print("\nbest single (%s): %.4f" % (names[best], singles[best]))

    seg = args.segment
    nseg = (H + seg - 1) // seg
    sel = np.zeros((C, nseg), dtype=np.int64)
    for c in range(C):
        for s in range(nseg):
            a, b = s * seg, min((s + 1) * seg, H)
            mses = [float(((V[e][:, c, a:b] - Yv[:, c, a:b]) ** 2).mean()) for e in range(E)]
            sel[c, s] = int(np.argmin(mses))

    routed = np.zeros_like(T[0])
    for c in range(C):
        for s in range(nseg):
            a, b = s * seg, min((s + 1) * seg, H)
            routed[:, c, a:b] = T[sel[c, s], :, c, a:b]
    print("per-segment routing: %.4f (%+.2f%%)" % (rmse(routed), (singles[best] - rmse(routed)) / singles[best] * 100))

    oracle = np.zeros_like(T[0])
    for c in range(C):
        for s in range(nseg):
            a, b = s * seg, min((s + 1) * seg, H)
            mses = [float(((T[e][:, c, a:b] - Y[:, c, a:b]) ** 2).mean()) for e in range(E)]
            oracle[:, c, a:b] = T[int(np.argmin(mses)), :, c, a:b]
    print("per-segment oracle: %.4f (%+.2f%%)" % (rmse(oracle), (singles[best] - rmse(oracle)) / singles[best] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
