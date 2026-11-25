"""Per-channel convex mixing of independently trained experts.

Instead of selecting one expert per channel, fit non-negative mixing weights on
the validation split and apply them to the test split.  Mixing can help even when
no single expert wins on a channel, as long as the experts' errors are
decorrelated.

  python tools/mix_routing.py --run-dirs A B ... --val-windows 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.routing_analysis import _make_ds, _drift
from training.core import get_model_instance
from training.baselines import make_repeated_seasonal_baseline_bcl


@torch.no_grad()
def predict_expert(run_dir: Path, val_windows: int):
    cfg = json.load(open(run_dir / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    test_ds = _make_ds(cfg, "test", train_ds)
    ti = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    pred_len = int(cfg["pred_len"])
    period = int(cfg.get("seasonal_period", 24))
    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=ti,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run_dir / "model_weights.pth", map_location=device))
    model.eval()
    name = list(model.expert_names)[0]

    def run(ds, n, batch_size=16):
        preds = []
        for start in range(0, n, batch_size):
            batch = [ds[i] for i in range(start, min(start + batch_size, n))]
            past = torch.stack([b["past"] for b in batch]).to(device)
            fut = torch.stack([b["future"] for b in batch]).to(device)
            kf = torch.stack([b["known_future"] for b in batch]).to(device)
            base = make_repeated_seasonal_baseline_bcl(past, pred_len, period)
            ctx = model._build_condition_context(past, known_future=kf)
            raw = model._align_tensor(past, model.num_channels, resize="none")
            expert = model.experts[0]
            ep = raw if getattr(expert, "uses_full_context", False) else ctx
            out = expert(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device), ep)
            pred = _drift((out + base).cpu().numpy(), past.cpu().numpy(), cfg, model.num_channels)
            preds.append(pred)
        return np.concatenate(preds, axis=0)  # [n, C, H]

    n_val = min(val_windows, len(val_ds))
    return name, run(val_ds, n_val), run(test_ds, len(test_ds)), test_ds


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-channel convex expert mixing.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--val-windows", type=int, default=1024)
    args = parser.parse_args(argv)

    names, Pv, Pt = [], [], []
    for d in args.run_dirs:
        name, v, t, _ = predict_expert(Path(d), args.val_windows)
        names.append(name)
        Pv.append(v)
        Pt.append(t)
        print(f"loaded {name}")

    V = np.stack(Pv)  # [E, nv, C, H]
    T = np.stack(Pt)  # [E, nt, C, H]
    E, nv, C, H = V.shape
    try:
        from scipy.optimize import nnls
        have_nnls = True
    except Exception:
        have_nnls = False

    # ground truth for validation (fit) and test (evaluation)
    cfg0 = json.load(open(Path(args.run_dirs[0]) / "config.json"))
    train_ds = _make_ds(cfg0, "train")
    val_ds = _make_ds(cfg0, "val", train_ds)
    test_ds = _make_ds(cfg0, "test", train_ds)
    n_fit = min(args.val_windows, len(val_ds))
    Yv = np.stack([val_ds[i]["future"].numpy() for i in range(n_fit)])  # [nv, C, H]
    Y = np.stack([test_ds[i]["future"].numpy() for i in range(len(test_ds))])  # [nt, C, H]

    def rmse(pred):
        return float(np.sqrt(((pred - Y) ** 2).mean()))

    single_preds = [T[e] for e in range(E)]
    singles = [rmse(p) for p in single_preds]
    best = int(np.argmin(singles))
    print(f"\nbest single ({names[best]}): {singles[best]:.4f}")

    equal = rmse(T.mean(axis=0))
    print(f"equal mix: {equal:.4f} ({(singles[best] - equal) / singles[best] * 100:+.2f}%)")

    if have_nnls:
        mix = np.zeros_like(T[0])
        for c in range(C):
            A = V[:, :, c, :].reshape(E, -1).T
            b = Yv[:, c, :].reshape(-1)
            w, _ = nnls(A, b)
            if w.sum() <= 0:
                w = np.zeros(E)
                w[best] = 1.0
            mix[:, c, :] = np.tensordot(w, T[:, :, c, :], axes=(0, 0))
            print("  channel %d: " % c + ", ".join("%s=%.2f" % (names[e], w[e]) for e in range(E) if w[e] > 0.01))
        print("nnls mix: %.4f (%+.2f%%)" % (rmse(mix), (singles[best] - rmse(mix)) / singles[best] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
