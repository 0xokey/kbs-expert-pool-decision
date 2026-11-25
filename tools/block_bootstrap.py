"""Block bootstrap for the per-window loss difference between two runs.

Windows overlap, so the i.i.d. bootstrap and the Wilcoxon test on windows are
anti-conservative.  This tool resamples contiguous blocks of windows instead and
reports a percentile interval for the mean loss difference and a one-sided
p-value.

  python tools/block_bootstrap.py --a <isolated run> --b <in-mixture run> --label ETTh1
"""
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


def block_bootstrap(a, b, block=336, boot=2000, seed=0):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b  # negative = A better
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(boot)
    for i in range(boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        means[i] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_one = float((means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), p_one


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--expert", default="DLinearDiffuser")
    p.add_argument("--label", default="")
    p.add_argument("--block", type=int, default=336)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = window_losses(args.a, args.expert, device, args.batch_size)
    b = window_losses(args.b, args.expert, device, args.batch_size)
    md, lo, hi, p = block_bootstrap(a, b, args.block, args.boot)
    rel = 100.0 * (-md) / float(b.mean())
    print(f"{args.label} n={min(len(a),len(b))} block={args.block} mean_diff={md:.5f} "
          f"CI95=[{lo:.5f},{hi:.5f}] tail_proportion={p:.4f} rel_gain={rel:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
