"""Evaluate a run with validation-selected per-channel routing.

Pipeline:
  1. score every expert on the validation split (scaled-space MSE, same protocol),
  2. pick the best expert per target channel,
  3. force that routing and run the regular evaluation (drift correction,
     scaling, CRPS disabled) on the test split.

    python tools/eval_fixed_routing.py --run-dir results/<run>
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

from datasets.generic_csv import GenericCSVDataset
from training.core import build_dataset, get_model_instance
from training.baselines import make_repeated_seasonal_baseline_bcl
from training.evaluation import run_evaluation
from utils.scheduler import DDPMScheduler


def _json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ds(cfg, mode, train_ds=None):
    kwargs = dict(
        seq_len=cfg["pred_len"], csv_path=cfg["csv_path"], mode=mode,
        context_len=cfg["context_len"], pred_len=cfg["pred_len"], channels_first=True,
        add_time_features=True, add_cyc_features=True, label_path=None,
        target_only_diffusion=True, split_borders=cfg.get("split_borders"),
        split_ratios=cfg.get("split_ratios"), scaler_type=cfg.get("scaler_type", "minmax"),
    )
    if train_ds is not None:
        kwargs["train_ds"] = train_ds
    return build_dataset(GenericCSVDataset, **kwargs)


@torch.no_grad()
def _expert_val_scores(model, ds, indices, target_idx, pred_len, period, device, batch_size=64):
    names = list(model.expert_names)
    scores = {name: [] for name in names}
    for start in range(0, len(indices), batch_size):
        batch = [ds[i] for i in indices[start:start + batch_size]]
        past = torch.stack([b["past"] for b in batch]).to(device)
        fut = torch.stack([b["future"] for b in batch]).to(device)
        kf = torch.stack([b["known_future"] for b in batch]).to(device)
        base = make_repeated_seasonal_baseline_bcl(past, pred_len, period)
        ctx = model._build_condition_context(past, known_future=kf)
        raw = model._align_tensor(past, model.num_channels, resize="none")
        true = fut.cpu().numpy().transpose(0, 2, 1)[:, :, target_idx]
        for i, name in enumerate(names):
            expert = model.experts[i]
            expert_past = raw if getattr(expert, "uses_full_context", False) else ctx
            out = expert(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device), expert_past)
            pred = (out + base).cpu().numpy().transpose(0, 2, 1)[:, :, target_idx]
            scores[name].append(((pred - true) ** 2).mean(axis=1))
    return {name: np.concatenate(v, axis=0) for name, v in scores.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate validation-selected routing.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    cfg = _json(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _ds(cfg, "train")
    val_ds = _ds(cfg, "val", train_ds)
    test_ds = _ds(cfg, "test", train_ds)
    target_idx = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    names = list(getattr(train_ds, "target_cols", [f"ch{i}" for i in target_idx]))
    pred_len = int(cfg["pred_len"])

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=target_idx,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()

    val_indices = list(range(min(args.val_windows, len(val_ds))))
    scores = _expert_val_scores(model, val_ds, val_indices, target_idx, pred_len,
                                int(cfg.get("seasonal_period", 24)), device)
    expert_names = list(model.expert_names)
    choice = []
    for t, name in enumerate(names):
        best = min(expert_names, key=lambda k: float(scores[k][:, t].mean()))
        choice.append(expert_names.index(best))
        print(f"{name:8s} -> {best:28s} val_rmse={float(np.sqrt(scores[best][:, t].mean())):.4f}")
    model.set_fixed_channel_experts(choice)

    scheduler = DDPMScheduler(
        num_timesteps=1000, device=device,
        schedule=cfg.get("noise_schedule", "linear"),
        prediction_type=str(cfg.get("prediction_type", "epsilon")),
    )
    ds = test_ds if args.split == "test" else val_ds
    metrics = run_evaluation(
        model, scheduler, ds, "FinalTestValBest", device, run,
        export_all_windows=False, max_eval_windows=len(ds),
        index_seed=int(cfg.get("seed", 42)) + 31, sample_seed=int(cfg.get("seed", 42)) + 2001,
        save_preview_plot=False, save_window_plots=False, save_overview_plots=False,
        sample_sampler=str(cfg.get("final_eval_sampler", "ddim")),
        sample_inference_steps=int(cfg.get("final_eval_inference_steps", 50)),
        sample_eta=float(cfg.get("final_eval_eta", 0.0)),
        sample_t_start=cfg.get("ddim_t_start"),
        use_amp=bool(cfg.get("use_amp", True)) and device.type == "cuda",
        drift_correction=bool(cfg.get("drift_correction", True)),
        drift_tail_window=int(cfg.get("drift_tail_window", 24)),
        drift_alpha=float(cfg.get("drift_alpha", 0.75)),
        horizon_boundaries=cfg.get("horizon_boundaries"),
        crps_samples=0,
        eval_batch_size=int(cfg.get("eval_batch_size", 32)),
        forecast_objective=str(cfg.get("forecast_objective", "direct")),
        seasonal_period=int(cfg.get("seasonal_period", 24)),
        baseline_reference=str(cfg.get("baseline_reference", "daily_seasonal")),
        eval_space=str(cfg.get("eval_space", "raw")),
        inference_mode=str(cfg.get("inference_mode", "direct")),
    )
    print(f"[ValBest] {args.split} RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
