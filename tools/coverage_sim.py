"""Simulation of the decision rule under known population risks.

Two error events are reported separately:
  - strict: a commit is wrong if the chosen expert is not strictly better than
    the runner-up in the population (ties count as errors);
  - set:    a commit is wrong if the chosen expert is not in the population
    argmin set (ties are allowed).
Fixed and sequential rules are compared on the same simulated losses and with
the same cap N.
"""
from __future__ import annotations

import argparse

import numpy as np


def decide(losses, c, idx, z):
    vec = losses[idx, c, :].mean(axis=0)
    order = np.argsort(vec)
    i1, i2 = order[0], order[1]
    s1 = losses[idx, c, i1].std(ddof=1)
    s2 = losses[idx, c, i2].std(ddof=1)
    n = len(idx)
    se = np.sqrt(s1 * s1 + s2 * s2) / np.sqrt(max(n, 2))
    margin = (vec[i2] - vec[i1]) / max(abs(vec[i1]), 1e-8)
    thr = z * se / max(abs(vec[i1]), 1e-8)
    return i1, i2, margin > thr


def make_problem(rng, C, K, N, sigma, delta):
    mu = np.zeros((C, K)); mu[:, 0] = -delta
    ties = rng.choice(C, size=max(1, C // 4), replace=False)
    mu[ties, 0] = 0.0; mu[ties, 1] = 0.0
    return mu, rng.normal(mu[None, :, :], sigma, size=(N, C, K))


def run(rng, mu, losses, C, K, N, z, sequential, n_fixed, n0, step):
    res = {"commits": 0, "strict_ok": 0, "set_ok": 0, "fw_strict": False, "fw_set": False}
    for c in range(C):
        if sequential:
            idx = rng.choice(N, size=n0, replace=False)
            while True:
                i1, i2, cert = decide(losses, c, idx, z)
                if cert or len(idx) >= N:
                    break
                remaining = np.setdiff1d(np.arange(N), idx)
                idx = np.union1d(idx, rng.choice(remaining, size=min(step, len(remaining)), replace=False))
        else:
            idx = rng.choice(N, size=n_fixed, replace=False)
            i1, i2, cert = decide(losses, c, idx, z)
        if not cert:
            continue
        res["commits"] += 1
        best = mu[c].min()
        strict = mu[c, i1] < mu[c, i2] - 1e-12
        in_set = abs(mu[c, i1] - best) < 1e-12
        res["strict_ok"] += int(strict)
        res["set_ok"] += int(in_set)
        if not strict: res["fw_strict"] = True
        if not in_set: res["fw_set"] = True
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--channels", type=int, default=21)
    p.add_argument("--experts", type=int, default=10)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--z", type=float, default=1.96)
    p.add_argument("--n0", type=int, default=32)
    p.add_argument("--step", type=int, default=32)
    p.add_argument("--capn", type=int, default=512)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    rows = [("fixed n=512", False, 512), ("sequential 32..512", True, 0)]
    aggs = {name: np.zeros(5) for name, _, _ in rows}
    for _ in range(args.reps):
        mu, losses = make_problem(rng, args.channels, args.experts, args.capn, args.sigma, args.delta)
        for name, seq, n in rows:
            r = run(rng, mu, losses, args.channels, args.experts, args.capn,
                    args.z, seq, n, args.n0, args.step)
            aggs[name] += [r["commits"], r["strict_ok"], r["set_ok"], int(r["fw_strict"]), int(r["fw_set"])]
    for name, _, _ in rows:
        c, s_ok, set_ok, fw_s, fw_set = aggs[name]
        print(f"{name:20s} commits/rep={c/args.reps:6.2f} strict_acc={s_ok/max(c,1):.3f} "
              f"set_acc={set_ok/max(c,1):.3f} FW_strict={fw_s/args.reps:.3f} FW_set={fw_set/args.reps:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
