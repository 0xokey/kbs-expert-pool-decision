"""Block bootstrap with several block lengths, reusing the per-window losses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pairwise_test import window_losses


def boot(d, block, boot_n, seed=0):
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(boot_n)
    for i in range(boot_n):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((means >= 0).mean())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--blocks", type=int, nargs="+", default=[192, 336, 672])
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--expert", default="DLinearDiffuser")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = window_losses(args.a, args.expert, dev, args.batch_size)
    b = window_losses(args.b, args.expert, dev, args.batch_size)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    for block in args.blocks:
        md, lo, hi, tail = boot(d, block, args.boot)
        rel = 100.0 * (-md) / float(b[:n].mean())
        print(f"{args.label} block={block:4d} mean_diff={md:.5f} CI95=[{lo:.5f},{hi:.5f}] "
              f"tail={tail:.4f} rel={rel:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
