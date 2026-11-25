"""Routing on time-aligned per-window losses (intersection of cutoffs)."""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def two_sample(x1, x2, z=1.96):
    n = len(x1)
    s1 = x1.std(ddof=1); s2 = x2.std(ddof=1)
    se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
    ref = max(abs(x1.mean()), 1e-8)
    return (x2.mean() - x1.mean()) / ref, z * se / ref


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--z", type=float, default=1.96)
    args = p.parse_args(argv)
    d = args.dir
    ds = args.dataset.lower()
    models = {}
    for f in glob.glob(os.path.join(d, "losses_official_dlinear_*.npz")):
        if ds in os.path.basename(f).lower():
            z = np.load(f)
            models["DLinear-official"] = (z["val"], z["val_cut"], z["test"], z["test_cut"])
    for f in glob.glob(os.path.join(d, "losses_*.npz")):
        b = os.path.basename(f).lower()
        if b.startswith("losses_official") or ds not in b:
            continue
        z = np.load(f)
        for i, n in enumerate(z["names"]):
            models[f"diff-{n}"] = (z["val"][i], z["val_cut"], z["test"][i], z["test_cut"])
    for f in glob.glob(os.path.join(d, "wloss_nf_*.npz")):
        b = os.path.basename(f).lower()
        if ds not in b:
            continue
        z = np.load(f)
        name = os.path.basename(f)[len("wloss_nf_"):].rsplit(f"_{ds}", 1)[0]
        models[f"nf-{name}"] = (z["val"], z["cut_val"], z["test"], z["cut_test"])
    names = list(models)
    common_val = None; common_test = None
    for n in names:
        v = set(models[n][1].tolist()); t = set(models[n][3].tolist())
        common_val = v if common_val is None else (common_val & v)
        common_test = t if common_test is None else (common_test & t)
    cv = np.array(sorted(common_val)); ct = np.array(sorted(common_test))
    def stack(which, cuts):
        arrs = []
        for n in names:
            m, mc = (models[n][0], models[n][1]) if which == "val" else (models[n][2], models[n][3])
            idx = np.searchsorted(mc, cuts)
            arrs.append(m[idx])
        return np.stack(arrs)
    val = stack("val", cv); test = stack("test", ct)
    M, C = val.shape[0], val.shape[2]
    vm = val.mean(1)
    order = np.argsort(vm, axis=0)
    best = int(np.argmin(vm.mean(1)))
    chosen = np.full(C, best, dtype=int)
    commits = 0
    for c in range(C):
        i1, i2 = int(order[0, c]), int(order[1, c])
        margin, thr = two_sample(val[i1, :, c], val[i2, :, c], args.z)
        if margin > thr:
            chosen[c] = i1; commits += 1
    def rmse(ix):
        return float(np.sqrt(np.mean([test[ix[c], :, c].mean() for c in range(C)])))
    print(f"{args.dataset}: models={M} ch={C} val_windows={len(cv)} test_windows={len(ct)} commits={commits}")
    print(f"  best single {names[best]} {rmse(np.full(C,best)):.4f}")
    print(f"  routed {rmse(chosen):.4f}  oracle {rmse(order[0,:]):.4f}")
    print(f"  used {len(set(chosen.tolist())|{best})}/{M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
