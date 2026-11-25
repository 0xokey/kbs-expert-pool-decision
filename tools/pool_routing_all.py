"""Certified routing over the full heterogeneous pool (DLinear + diffusion + NF)."""
from __future__ import annotations

import argparse
import glob
import os

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
    p.add_argument("--dir", required=True)
    p.add_argument("--z", type=float, default=1.96)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    d = args.dir
    models = {}
    for f in sorted(glob.glob(os.path.join(d, "losses_official_dlinear_*.npz"))):
        if args.dataset.lower() in os.path.basename(f).lower():
            z = np.load(f)
            models["official-DLinear"] = (z["val"], z["test"])
    for f in glob.glob(os.path.join(d, "losses_*.npz")):
        b = os.path.basename(f).lower()
        if b.startswith("losses_official"):
            continue
        if args.dataset.lower() == "weather" and "weather" in b:
            z = np.load(f)
            for i, n in enumerate(z["names"]):
                models[str(n)] = (z["val"][i], z["test"][i])
        elif args.dataset.lower() == "etth1" and b.startswith("losses_etth1"):
            z = np.load(f)
            for i, n in enumerate(z["names"]):
                models[str(n)] = (z["val"][i], z["test"][i])
        elif args.dataset.lower() == "ettm1" and b.startswith("losses_ettm1"):
            z = np.load(f)
            for i, n in enumerate(z["names"]):
                models[str(n)] = (z["val"][i], z["test"][i])
    for f in sorted(glob.glob(os.path.join(d, "nf_*.npz"))):
        b = os.path.basename(f).lower()
        if args.dataset.lower() not in b:
            continue
        if f"_s{args.seed}.npz" not in b:
            continue
        z = np.load(f)
        name = b.replace("nf_", "").replace(f"_{args.dataset.lower()}.npz", "")
        models[f"nf-{name}"] = (z["val"], z["test"])
    names = list(models)
    nv = min(models[n][0].shape[0] for n in names)
    nt = min(models[n][1].shape[0] for n in names)
    val = np.stack([models[n][0][:nv] for n in names])
    test = np.stack([models[n][1][:nt] for n in names])
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
    def rmse(models_idx):
        return float(np.sqrt(np.mean([test[models_idx[c], :, c].mean() for c in range(C)])))
    print(f"{args.dataset}: pool={M}, channels={C}, committed={committed}")
    print(f"  best single = {names[best_single]}  test={rmse(np.full(C, best_single)):.4f}")
    print(f"  routed  test={rmse(chosen):.4f}  oracle={rmse(order[0, :]):.4f}")
    used = sorted(set(chosen.tolist()) | {best_single})
    print(f"  used {len(used)}/{M}: {[names[i] for i in used]}")
    ranking = sorted(range(M), key=lambda i: vm[i].mean())
    print("  top-5 by val:", [(names[i], round(float(np.sqrt(test[i].mean())), 4)) for i in ranking[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
