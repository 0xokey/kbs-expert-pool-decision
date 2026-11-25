"""Unified routing analysis: gate vs single experts vs val-best vs oracle.

All numbers are produced by the *same* evaluation function (drift correction,
scaled metrics, direct inference), so they are directly comparable.

    python tools/routing_analysis.py --run-dir results/<run> --split test

Outputs a JSON + CSV next to the run directory.
"""
from __future__ import annotations

import argparse
import csv
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
from utils.postprocess import apply_linear_drift_correction


def _json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_ds(cfg, mode, train_ds=None):
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


def _to_bcl(x: np.ndarray, channels: int) -> np.ndarray:
    return x if x.shape[1] == channels else x.transpose(0, 2, 1)


def _drift(pred_bcl, past_bcl, cfg, channels):
    if not bool(cfg.get("drift_correction", True)):
        return pred_bcl
    pred_bcl = _to_bcl(np.asarray(pred_bcl), channels)
    past_bcl = _to_bcl(np.asarray(past_bcl), channels)
    assert pred_bcl.shape[1] == channels, pred_bcl.shape
    assert past_bcl.shape[1] == channels, past_bcl.shape
    out = np.empty_like(pred_bcl)
    for i in range(pred_bcl.shape[0]):
        p = pred_bcl[i].T
        h = past_bcl[i].T
        out[i] = apply_linear_drift_correction(
            h, p, target_indices=list(range(channels)),
            tail_window=int(cfg.get("drift_tail_window", 24)),
            alpha=float(cfg.get("drift_alpha", 0.75)),
        ).T
    return out


@torch.no_grad()
def _expert_channel_scores(model, ds, indices, target_idx, pred_len, period, cfg, device, batch_size=64):
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
            pred = _drift((out + base).cpu().numpy(), past.cpu().numpy(), cfg, model.num_channels)
            pred = pred.transpose(0, 2, 1)[:, :, target_idx]
            scores[name].append(((pred - true) ** 2).mean(axis=1))
    return {name: np.concatenate(v, axis=0) for name, v in scores.items()}


@torch.no_grad()
def _routed_channel_scores(model, ds, indices, target_idx, pred_len, period, cfg, device, batch_size=64):
    rows = []
    for start in range(0, len(indices), batch_size):
        batch = [ds[i] for i in indices[start:start + batch_size]]
        past = torch.stack([b["past"] for b in batch]).to(device)
        fut = torch.stack([b["future"] for b in batch]).to(device)
        kf = torch.stack([b["known_future"] for b in batch]).to(device)
        cl = torch.stack([b.get("coarse_label", torch.zeros(ds.num_channels, dtype=torch.long)) for b in batch]).to(device)
        base = make_repeated_seasonal_baseline_bcl(past, pred_len, period)
        out = model(torch.zeros_like(fut), torch.zeros(fut.size(0), dtype=torch.long, device=device),
                    past, coarse_label=cl, known_future=kf, return_weights=False)
        pred = _drift((out + base).cpu().numpy(), past.cpu().numpy(), cfg, model.num_channels)
        pred = pred.transpose(0, 2, 1)[:, :, target_idx]
        true = fut.cpu().numpy().transpose(0, 2, 1)[:, :, target_idx]
        rows.append(((pred - true) ** 2).mean(axis=1))
    return np.concatenate(rows, axis=0)


def _rmse_from_scores(scores):
    return float(np.sqrt(float(np.mean(scores))))


def _evaluate(model, ds, cfg, device, epoch_name):
    scheduler = DDPMScheduler(
        num_timesteps=1000, device=device, schedule=cfg.get("noise_schedule", "linear"),
        prediction_type=str(cfg.get("prediction_type", "epsilon")),
    )
    metrics = run_evaluation(
        model, scheduler, ds, epoch_name, device, Path(cfg["_run_dir"]),
        export_all_windows=False, max_eval_windows=len(ds),
        index_seed=int(cfg.get("seed", 42)) + 31, sample_seed=int(cfg.get("seed", 42)) + 2001,
        save_preview_plot=False, save_window_plots=False, save_overview_plots=False,
        sample_sampler=str(cfg.get("final_eval_sampler", "ddim")),
        sample_inference_steps=int(cfg.get("final_eval_inference_steps", 50)),
        sample_eta=float(cfg.get("final_eval_eta", 0.0)), sample_t_start=cfg.get("ddim_t_start"),
        use_amp=bool(cfg.get("use_amp", True)) and device.type == "cuda",
        drift_correction=bool(cfg.get("drift_correction", True)),
        drift_tail_window=int(cfg.get("drift_tail_window", 24)),
        drift_alpha=float(cfg.get("drift_alpha", 0.75)),
        horizon_boundaries=cfg.get("horizon_boundaries"), crps_samples=0,
        eval_batch_size=int(cfg.get("eval_batch_size", 32)),
        forecast_objective=str(cfg.get("forecast_objective", "direct")),
        seasonal_period=int(cfg.get("seasonal_period", 24)),
        baseline_reference=str(cfg.get("baseline_reference", "daily_seasonal")),
        eval_space=str(cfg.get("eval_space", "scaled")),
        inference_mode=str(cfg.get("inference_mode", "direct")),
    )
    return float(metrics["RMSE"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified routing analysis.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--eval-windows", type=int, default=None)
    parser.add_argument("--skip-single", action="store_true")
    args = parser.parse_args(argv)

    run = Path(args.run_dir).resolve()
    cfg = _json(run / "config.json")
    cfg["_run_dir"] = str(run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = _make_ds(cfg, "train")
    val_ds = _make_ds(cfg, "val", train_ds)
    test_ds = _make_ds(cfg, "test", train_ds)
    target_idx = list(getattr(train_ds, "target_indices", range(train_ds.num_channels)))
    names = list(getattr(train_ds, "target_cols", [f"ch{i}" for i in target_idx]))
    pred_len = int(cfg["pred_len"])
    period = int(cfg.get("seasonal_period", 24))

    model = get_model_instance(cfg, train_ds.num_channels, device, target_indices=target_idx,
                               known_future_channels=train_ds.known_future_channels)
    model.load_state_dict(torch.load(run / "model_weights.pth", map_location=device))
    model.eval()
    expert_names = list(model.expert_names)

    eval_ds = test_ds if args.split == "test" else val_ds

    val_indices = list(range(min(args.val_windows, len(val_ds))))
    scores = _expert_channel_scores(model, val_ds, val_indices, target_idx, pred_len, period, cfg, device)
    half_a = val_indices[::2]
    half_b = val_indices[1::2]
    scores_a = _expert_channel_scores(model, val_ds, half_a, target_idx, pred_len, period, cfg, device) if len(half_a) > 8 else None
    scores_b = _expert_channel_scores(model, val_ds, half_b, target_idx, pred_len, period, cfg, device) if len(half_b) > 8 else None
    test_scores = _expert_channel_scores(model, eval_ds, list(range(len(eval_ds))), target_idx, pred_len, period, cfg, device)

    results = {"run": run.name, "split": args.split, "channels": names, "experts": expert_names}
    eval_indices = list(range(len(eval_ds)))
    model.clear_fixed_channel_experts()
    results["gate"] = _rmse_from_scores(
        _routed_channel_scores(model, eval_ds, eval_indices, target_idx, pred_len, period, cfg, device)
    )

    if not args.skip_single:
        singles = {}
        for i, name in enumerate(expert_names):
            model.set_fixed_channel_experts([i] * len(target_idx))
            singles[name] = _rmse_from_scores(
                _routed_channel_scores(model, eval_ds, eval_indices, target_idx, pred_len, period, cfg, device)
            )
        results["single"] = singles
        best_single = min(singles, key=singles.get)
        results["best_single"] = {"expert": best_single, "rmse": singles[best_single]}

    val_choice, oracle_choice = [], []
    per_channel = []
    margins = []
    for t, ch in enumerate(names):
        val_vec = np.array([float(scores[k][:, t].mean()) for k in expert_names])
        order = np.argsort(val_vec)
        best_val = float(val_vec[order[0]])
        second_val = float(val_vec[order[1]]) if len(order) > 1 else best_val
        margins.append((second_val - best_val) / max(best_val, 1e-8))
        val_best = expert_names[int(order[0])]
        test_best = min(expert_names, key=lambda k: float(test_scores[k][:, t].mean()))
        val_choice.append(expert_names.index(val_best))
        oracle_choice.append(expert_names.index(test_best))
        per_channel.append({
            "channel": ch,
            "val_best": val_best,
            "oracle_best": test_best,
            "val_rmse_best": float(np.sqrt(scores[val_best][:, t].mean())),
            "oracle_rmse_best": float(np.sqrt(test_scores[test_best][:, t].mean())),
        })
    results["per_channel"] = per_channel
    # Median is robust to channels whose best expert is near-perfect (the mean
    # would be dominated by such outliers).
    results["selection_margin"] = float(np.median(margins))

    model.set_fixed_channel_experts(val_choice)
    results["val_best_routing"] = _rmse_from_scores(
        _routed_channel_scores(model, eval_ds, eval_indices, target_idx, pred_len, period, cfg, device)
    )
    model.set_fixed_channel_experts(oracle_choice)
    results["oracle_routing"] = _rmse_from_scores(
        _routed_channel_scores(model, eval_ds, eval_indices, target_idx, pred_len, period, cfg, device)
    )

    # Spread and preference stability are the two quantities the paper's
    # criterion is built on.
    if "single" in results:
        single_values = list(results["single"].values())
        best_value = min(single_values)
        results["spread"] = float((float(np.mean(single_values)) - best_value) / max(best_value, 1e-8))
    if scores_a is not None and scores_b is not None:
        agree = 0
        rho = []
        for t in range(len(names)):
            vec_a = np.array([float(scores_a[k][:, t].mean()) for k in expert_names])
            vec_b = np.array([float(scores_b[k][:, t].mean()) for k in expert_names])
            agree += int(int(np.argmin(vec_a)) == int(np.argmin(vec_b)))
            rank_a = np.argsort(np.argsort(vec_a))
            rank_b = np.argsort(np.argsort(vec_b))
            rho.append(float(np.corrcoef(rank_a, rank_b)[0, 1]) if np.std(rank_a) > 0 and np.std(rank_b) > 0 else 0.0)
        results["preference_stability"] = float(agree / max(len(names), 1))
        results["rank_stability"] = float(np.nanmean(rho))
    if "best_single" in results:
        best_value = float(results["best_single"]["rmse"])
        results["gain_val_best"] = float((best_value - results["val_best_routing"]) / max(best_value, 1e-8))
        results["gain_oracle"] = float((best_value - results["oracle_routing"]) / max(best_value, 1e-8))
        results["gain_gate"] = float((best_value - results["gate"]) / max(best_value, 1e-8))
    out_json = run / f"routing_analysis_{args.split}.json"
    out_csv = run / f"routing_analysis_{args.split}.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    rows = [{"method": "gate", "rmse": results["gate"]}]
    if "single" in results:
        for k, v in results["single"].items():
            rows.append({"method": f"single:{k}", "rmse": v})
    rows.append({"method": "val_best_routing", "rmse": results["val_best_routing"]})
    rows.append({"method": "oracle_routing", "rmse": results["oracle_routing"]})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rmse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {run.name} ({args.split}) ===")
    for row in rows:
        print(f"  {row['method']:34s} {row['rmse']:.4f}")
    if "spread" in results:
        print(f"  spread={results['spread']:.3f} margin={results.get('selection_margin', float('nan')):.3f} "
              f"stability={results.get('preference_stability', float('nan')):.2f} "
              f"rank_stability={results.get('rank_stability', float('nan')):.2f} "
              f"gain(val_best)={results.get('gain_val_best', float('nan')):+.3f} "
              f"gain(oracle)={results.get('gain_oracle', float('nan')):+.3f} "
              f"gain(gate)={results.get('gain_gate', float('nan')):+.3f}")
    print(f"written -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
