"""Certified routing over a heterogeneous pool of independently trained models.

Inputs are per-window, per-channel losses:
  - a .npz with arrays val and test of shape [n, C] for the official DLinear;
  - a .npz with arrays val [E, n_val, C] and test [E, n_test, C] for the
    diffusion experts, plus the expert names.
"""
from __future__ import annotations

import argparse

import numpy as np


def two_sample(x1, x2, z=1.96):
    n = min(len(x1), len(x2))
    x1, x2 = x1[:n], x2[:n]
    s1 = x1.std(ddof=1); s2 = x2.std(ddof=1)
    se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
    ref = max(abs(x1.mean()), 1e-8)
    return (x2.mean() - x1.mean()) / ref, z * se / ref


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--dlinear", required=True)
    p.add_argument("--experts", required=True)
    p.add_argument("--z", type=float, default=1.96)
    args = p.parse_args(argv)
    dl = np.load(args.dlinear)
    ex = np.load(args.experts)
    names = ["official-DLinear"] + [str(n) for n in ex["names"]]
    n_val = min(dl["val"].shape[0], ex["val"].shape[1])
    n_test = min(dl["test"].shape[0], ex["test"].shape[1])
    val = np.concatenate([dl["val"][None, :n_val], ex["val"][:, :n_val]], axis=0)
    test = np.concatenate([dl["test"][None, :n_test], ex["test"][:, :n_test]], axis=0)
    M, _, C = val.shape
    vm = val.mean(axis=1)
    order = np.argsort(vm, axis=0)
    best_single = int(np.argmin(vm.mean(axis=1)))
    chosen = np.full(C, best_single, dtype=int)
    committed = 0
    for c in range(C):
        i1, i2 = int(order[0, c]), int(order[1, c])
        margin, thr = two_sample(val[i1, :, c], val[i2, :, c], args.z)
        if margin > thr:
            chosen[c] = i1
            committed += 1
    def rmse(models):
        return float(np.sqrt(np.mean([test[models[c], :, c].mean() for c in range(C)])))
    routed = rmse(chosen)
    single = rmse(np.full(C, best_single))
    oracle = rmse(order[0, :])
    used = sorted(set(chosen.tolist()) | {best_single})
    print(f"{args.dataset}: pool={M} models, channels={C}, committed={committed}")
    print(f"  best single = {names[best_single]}  test={single:.4f}")
    print(f"  routed  test={routed:.4f}  oracle={oracle:.4f}  used={len(used)}/{M}")
    print(f"  used: {[names[i] for i in used]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
