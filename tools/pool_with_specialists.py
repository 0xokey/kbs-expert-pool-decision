"""Routing with complementary channel-group specialists added to the pool."""
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z", type=float, default=1.96)
    args = p.parse_args(argv)
    d = args.dir
    models = {}
    for f in sorted(glob.glob(os.path.join(d, "losses_official_dlinear_*.npz"))):
        if args.dataset.lower() in os.path.basename(f).lower():
            z = np.load(f); models["official-DLinear"] = (z["val"], z["test"])
    for f in glob.glob(os.path.join(d, "losses_*.npz")):
        b = os.path.basename(f).lower()
        if b.startswith("losses_official"):
            continue
        if args.dataset.lower() in b and not b.startswith("losses_official"):
            z = np.load(f)
            for i, n in enumerate(z["names"]):
                models[str(n)] = (z["val"][i], z["test"][i])
    for f in sorted(glob.glob(os.path.join(d, "nf_*.npz"))):
        b = os.path.basename(f).lower()
        if args.dataset.lower() not in b or f"_s{args.seed}.npz" not in b:
            continue
        z = np.load(f)
        name = os.path.basename(f).replace("nf_", "").rsplit(f"_{args.dataset.lower()}", 1)[0]
        models[f"nf-{name}"] = (z["val"], z["test"])
    spec = None
    for f in sorted(glob.glob(os.path.join(d, f"spec_*_{args.dataset.lower()}_g*_s{args.seed}.npz"))):
        z = np.load(f)
        groups = z["groups"].ravel()
        spec = {"val": z["val"], "test": z["test"], "groups": groups,
                "name": os.path.basename(f).replace("spec_", "spec-").replace(".npz", "")}
    names = list(models)
    nv = min(models[n][0].shape[0] for n in names)
    nt = min(models[n][1].shape[0] for n in names)
    if spec is not None:
        nv = min(nv, spec["val"].shape[0]); nt = min(nt, spec["test"].shape[0])
    C = models[names[0]][0].shape[1]
    vm = np.stack([models[n][0][:nv] for n in names]).mean(1)
    tm = np.stack([models[n][1][:nt] for n in names]).mean(1)
    best_global = int(np.argmin(vm.mean(1)))

    def run(use_spec):
        chosen = np.full(C, best_global, dtype=int)
        for c in range(C):
            cand_v = [vm[i, c] for i in range(len(names))]
            cand_t = [tm[i, c] for i in range(len(names))]
            cand_w = [models[n][0][:nv, c] for n in names]
            cand_wt = [models[n][1][:nt, c] for n in names]
            label = list(names)
            if use_spec:
                g = int(spec["groups"][c])
                cand_v.append(float(spec["val"][:nv, c].mean()))
                cand_t.append(float(spec["test"][:nt, c].mean()))
                cand_w.append(spec["val"][:nv, c])
                cand_wt.append(spec["test"][:nt, c])
                label = label + [f"{spec['name']}-g{g}"]
            order = np.argsort(cand_v)
            i1, i2 = int(order[0]), int(order[1])
            margin, thr = two_sample(cand_w[i1], cand_w[i2], args.z)
            if margin > thr:
                chosen[c] = i1
        rmse = float(np.sqrt(np.mean([tm[chosen[c], c] if chosen[c] < len(names)
                                      else spec_pick(chosen[c], c) for c in range(C)])))
        return chosen, rmse

    def spec_pick(idx, c):
        return float(spec["test"][:nt, c].mean()) if spec is not None else float("nan")

    chosen_s, rmse_s = run(True)
    chosen_g, rmse_g = run(False)
    print(f"{args.dataset} s{args.seed}: best global {names[best_global]} {tm[best_global].mean()**0.5:.4f}")
    print(f"  global-only routed {rmse_g:.4f}")
    print(f"  with specialists  {rmse_s:.4f}  (specialists used: {sum(1 for v in chosen_s if v >= len(names))}/{C})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
