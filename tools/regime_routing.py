"""Regime-conditioned routing over a pool of independently trained experts.

The coarse regime label of a window is derived from its *history* (trend /
seasonality / volatility), so it is available at inference time.  For each
channel and regime we select, on validation, the expert with the lowest mean
loss in that regime, and apply the choice to the test windows of the same
regime.

Reported numbers:
  best single      : one expert for all channels/regimes (validation-selected)
  static oracle    : per channel, test-best expert
  regime oracle    : per channel and regime, test-best expert (upper bound)
  regime routing   : per channel and regime, validation-best expert

Usage:
  python tools/regime_routing.py --run-dirs A B ... --dataset ETTh1 --label-dir labels_derived
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

from tools.pool_routing import load_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regime-conditioned routing.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--dataset", required=True, help="label base name, e.g. ETTh1")
    parser.add_argument("--label-dir", default="labels_derived")
    parser.add_argument("--val-windows", type=int, default=1024)
    parser.add_argument("--min-per-regime", type=int, default=20)
    args = parser.parse_args(argv)

    names, vals, tests = [], [], []
    for d in args.run_dirs:
        name, v, t = load_scores(Path(d), args.val_windows)
        names.append(name)
        vals.append(v)
        tests.append(t)
        print(f"loaded {name}")

    V = np.stack(vals)   # [E, n_val, C]
    T = np.stack(tests)  # [E, n_test, C]
    E, C = V.shape[0], V.shape[2]

    val_labels = np.load(Path(args.label_dir) / f"{args.dataset}_val_labels.npy")
    test_labels = np.load(Path(args.label_dir) / f"{args.dataset}_test_labels.npy")
    if val_labels.ndim == 1:
        val_labels = np.repeat(val_labels[:, None], C, axis=1)
    if test_labels.ndim == 1:
        test_labels = np.repeat(test_labels[:, None], C, axis=1)
    n_val = V.shape[1]
    val_labels = val_labels[:n_val]

    val_mse = V.mean(axis=1)
    test_mse = T.mean(axis=1)
    best_single = int(np.argmin(val_mse.mean(axis=1)))
    single = float(np.sqrt(test_mse[best_single].mean()))
    oracle = float(np.sqrt(test_mse.min(axis=0).mean()))

    # static val-best
    static_pick = np.argmin(val_mse, axis=0)
    static_val = float(np.sqrt(test_mse[static_pick, np.arange(C)].mean()))

    # regime-conditioned
    regimes = np.unique(val_labels)
    pick = np.full((C, len(regimes)), best_single, dtype=np.int64)
    for ci, r in enumerate(regimes):
        mask = val_labels == r
        if mask.sum() < args.min_per_regime:
            continue
        per_expert = np.array([V[e][mask].mean(axis=0) for e in range(E)])  # [E, C]
        pick[:, ci] = np.argmin(per_expert, axis=0)

    regime_index = {int(r): i for i, r in enumerate(regimes)}
    num = 0.0
    cnt = 0
    for w in range(T.shape[1]):
        for c in range(C):
            ri = regime_index.get(int(test_labels[w, c]), None)
            e = pick[c, ri] if ri is not None else best_single
            num += T[e, w, c]
            cnt += 1
    regime_rmse = float(np.sqrt(num / max(cnt, 1)))

    # regime oracle
    num_o, cnt_o = 0.0, 0
    for w in range(T.shape[1]):
        for c in range(C):
            num_o += T[:, w, c].min()
            cnt_o += 1
    regime_oracle = float(np.sqrt(num_o / max(cnt_o, 1)))

    print(f"\nE={E} C={C} regimes={len(regimes)}")
    print(f"best single ({names[best_single]}): {single:.4f}")
    print(f"static val-best: {static_val:.4f} ({(single - static_val) / single * 100:+.2f}%)")
    print(f"static oracle: {oracle:.4f} ({(single - oracle) / single * 100:+.2f}%)")
    print(f"regime routing: {regime_rmse:.4f} ({(single - regime_rmse) / single * 100:+.2f}%)")
    print(f"regime oracle: {regime_oracle:.4f} ({(single - regime_oracle) / single * 100:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
