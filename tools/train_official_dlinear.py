"""Official LTSF-Linear DLinear under this project's protocol (cached tensors)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from tools.routing_analysis import _make_ds, _drift
from training.baselines import make_repeated_seasonal_baseline_bcl
from dlinear import Model as DLinear


def materialise(ds, key):
    path = ROOT / "results" / f"cache_{key}.npz"
    if path.exists():
        z = np.load(path)
        return torch.from_numpy(z["p"]), torch.from_numpy(z["f"])
    past, fut = [], []
    for i in range(len(ds)):
        b = ds[i]
        past.append(b["past"]); fut.append(b["future"])
    P, F = torch.stack(past), torch.stack(fut)
    np.savez_compressed(path, p=P.numpy(), f=F.numpy())
    return P, F


@torch.no_grad()
def evaluate(model, past, fut, cfg, device, pred_len, period, bs=256):
    model.eval()
    outs = []
    for s in range(0, len(past), bs):
        p = past[s:s+bs].to(device)
        base = make_repeated_seasonal_baseline_bcl(p, pred_len, period).cpu()
        out = model(p.permute(0, 2, 1)).permute(0, 2, 1).cpu() + base
        outs.append(out.numpy())
    pred = np.concatenate(outs, 0)
    hist = past.numpy()
    if bool(cfg.get("drift_correction", True)):
        k = int(cfg.get("drift_tail_window", 24))
        alpha = float(cfg.get("drift_alpha", 0.75))
        k = max(1, min(k, hist.shape[2], pred.shape[1]))
        hist_level = hist[:, :, -k:].mean(axis=2, keepdims=True)
        pred_level = pred[:, :, :k].mean(axis=2, keepdims=True)
        decay = np.linspace(1.0, 0.0, num=pred.shape[1], dtype=np.float32).reshape(1, -1, 1)
        pred = pred + alpha * decay * (hist_level - pred_level)
    se = (pred - fut.numpy()) ** 2  # [n, C, pred_len]
    per_window = se.mean(axis=2)    # [n, C]
    return float(np.sqrt(se.mean())), per_window


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    cfg = json.load(open(args.config))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = _make_ds(cfg, "train"); va = _make_ds(cfg, "val", tr); te = _make_ds(cfg, "test", tr)
    C = tr.num_channels
    pl = int(cfg["pred_len"]); sl = int(cfg["context_len"]); per = int(cfg.get("seasonal_period", 24))
    def cutoffs(ds):
        out = []
        for i in range(len(ds)):
            hist, _ = ds.get_time_window(i)
            h = hist[-1]
            out.append(np.datetime64(h, "ns").astype("int64") if isinstance(h, (np.datetime64, str)) else int(h))
        return np.array(out, dtype=np.int64)
    stem = Path(cfg["csv_path"]).stem
    Ptr, Ftr = materialise(tr, f"{stem}_train_{sl}_{pl}")
    Pva, Fva = materialise(va, f"{stem}_val_{sl}_{pl}")
    Pte, Fte = materialise(te, f"{stem}_test_{sl}_{pl}")
    model_cfg = SimpleNamespace(seq_len=sl, pred_len=pl, individual=False, enc_in=C, dec_in=C, c_out=C)
    model = DLinear(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Ptr), generator=torch.Generator().manual_seed(args.seed + ep))
        for s in range(0, len(Ptr), args.batch_size):
            idx = perm[s:s+args.batch_size]
            past = Ptr[idx].to(device); fut = Ftr[idx].to(device)
            base = make_repeated_seasonal_baseline_bcl(past, pl, per).to(device)
            target = fut - base
            opt.zero_grad()
            pred = model(past.permute(0, 2, 1)).permute(0, 2, 1)
            loss = loss_fn(pred, target)
            loss.backward(); opt.step()
        val, _ = evaluate(model, Pva, Fva, cfg, device, pl, per)
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test, test_pw = evaluate(model, Pte, Fte, cfg, device, pl, per)
    _, val_pw = evaluate(model, Pva, Fva, cfg, device, pl, per)
    np.savez_compressed(ROOT / "results" / f"losses_official_dlinear_{Path(cfg['csv_path']).stem}.npz",
                        val=val_pw, test=test_pw,
                        val_cut=cutoffs(va), test_cut=cutoffs(te))
    out = {"dataset": Path(cfg["csv_path"]).stem, "seed": args.seed, "val_rmse": best,
           "test_rmse": test, "epochs_run": ep + 1}
    print(json.dumps(out))
    dest = ROOT / "results" / f"official_dlinear_{out['dataset']}_s{args.seed}.json"
    json.dump(out, open(dest, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
