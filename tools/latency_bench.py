"""Inference latency of the full pool versus the pruned pool."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.routing_analysis import _make_ds
from training.core import get_model_instance


def bench(run_dir, device, n=256, iters=10, batch_size=32):
    run = Path(run_dir)
    cfg = json.load(open(run / "config.json"))
    ds = _make_ds(cfg, "test")
    ti = list(getattr(ds, "target_indices", range(ds.num_channels)))
    model = get_model_instance(cfg, ds.num_channels, device, target_indices=ti,
                               known_future_channels=ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    batches = []
    for start in range(0, n, batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, n))]
        past = torch.stack([b["past"] for b in batch]).to(device)
        fut = torch.stack([b["future"] for b in batch]).to(device)
        kf = torch.stack([b["known_future"] for b in batch]).to(device)
        cl = torch.stack([b.get("coarse_label", torch.zeros(ds.num_channels, dtype=torch.long)) for b in batch]).to(device)
        batches.append((past, fut, kf, cl))

    def run(b):
        past, fut, kf, cl = b
        return model(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device),
                     past, coarse_label=cl, known_future=kf, return_weights=False)

    with torch.no_grad():
        for _ in range(2):
            for b in batches:
                run(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for b in batches:
                run(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    windows = sum(len(b[0]) for b in batches) * iters
    return dt / windows * 1000.0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True)
    p.add_argument("--pruned", required=True)
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = bench(args.full, device, n=args.n)
    pruned = bench(args.pruned, device, n=args.n)
    print(f"full_ms_per_window={full:.3f} pruned_ms_per_window={pruned:.3f} speedup={full/pruned:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
