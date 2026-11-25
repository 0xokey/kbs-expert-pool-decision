"""Train an official PatchTST or iTransformer model under this project's protocol."""
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


def build(model_name, C, sl, pl, freq="h"):
    if model_name == "patchtst":
        from patchtst import Model
        cfg = SimpleNamespace(enc_in=C, seq_len=sl, pred_len=pl, e_layers=3, n_heads=8,
                              d_model=128, d_ff=256, dropout=0.1, fc_dropout=0.1,
                              head_dropout=0.0, individual=False, patch_len=16, stride=8,
                              padding_patch="end", revin=1, affine=0, subtract_last=0,
                              decomposition=0, kernel_size=25)
        return Model(cfg)
    if model_name == "itransformer":
        from itransformer import Model
        cfg = SimpleNamespace(enc_in=C, seq_len=sl, pred_len=pl, output_attention=False,
                              use_norm=1, d_model=128, embed="timeF", freq=freq,
                              dropout=0.1, class_strategy="projection", e_layers=2,
                              n_heads=8, d_ff=256, factor=1, activation="gelu")
        return Model(cfg)
    raise ValueError(model_name)


def forward(model_name, model, past_bcl):
    x = past_bcl.permute(0, 2, 1)
    if model_name == "itransformer":
        out = model(x, None, None, None)
    else:
        out = model(x)
    return out.permute(0, 2, 1)


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
def evaluate(name, model, past, fut, cfg, device, pl, per, bs=128):
    model.eval()
    outs = []
    for s in range(0, len(past), bs):
        p = past[s:s+bs].to(device)
        base = make_repeated_seasonal_baseline_bcl(p, pl, per).cpu()
        outs.append((forward(name, model, p).cpu() + base).numpy())
    pred = np.concatenate(outs, 0)
    pred = _drift(pred, past.numpy(), cfg, pred.shape[1])
    return float(np.sqrt(((pred - fut.numpy()) ** 2).mean()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["patchtst", "itransformer"])
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args(argv)
    cfg = json.load(open(args.config))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = _make_ds(cfg, "train"); va = _make_ds(cfg, "val", tr); te = _make_ds(cfg, "test", tr)
    C = tr.num_channels; pl = int(cfg["pred_len"]); sl = int(cfg["context_len"])
    per = int(cfg.get("seasonal_period", 24))
    stem = Path(cfg["csv_path"]).stem
    Ptr, Ftr = materialise(tr, f"{stem}_train_{sl}_{pl}")
    Pva, Fva = materialise(va, f"{stem}_val_{sl}_{pl}")
    Pte, Fte = materialise(te, f"{stem}_test_{sl}_{pl}")
    model = build(args.model, C, sl, pl).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Ptr), generator=torch.Generator().manual_seed(args.seed + ep))
        for s in range(0, len(Ptr), args.batch_size):
            idx = perm[s:s+args.batch_size]
            past = Ptr[idx].to(dev); fut = Ftr[idx].to(dev)
            base = make_repeated_seasonal_baseline_bcl(past, pl, per).to(dev)
            opt.zero_grad()
            loss = loss_fn(forward(args.model, model, past), fut - base)
            loss.backward(); opt.step()
        val = evaluate(args.model, model, Pva, Fva, cfg, dev, pl, per)
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(args.model, model, Pte, Fte, cfg, dev, pl, per)
    out = {"model": args.model, "dataset": stem, "seed": args.seed,
           "val_rmse": best, "test_rmse": test, "epochs_run": ep + 1}
    print(json.dumps(out))
    dest = ROOT / "results" / f"official_{args.model}_{stem}_s{args.seed}.json"
    json.dump(out, open(dest, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
