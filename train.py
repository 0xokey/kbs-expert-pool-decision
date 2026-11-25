import os
import csv
import contextlib
import datetime
import json
import joblib
import argparse
import inspect
import random
import sys
import time
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from utils.scheduler import DDPMScheduler
from utils.visualizer import save_prediction_plot, save_series_overview_plot
from utils.metrics import MetricsCalculator, resolve_horizon_ranges
from utils.postprocess import apply_linear_drift_correction
from utils.run_logging import setup_run_log
from utils.common import instantiate_model, resolve_context_pred_lengths, resolve_label_path
from utils.config import validate_config

from datasets.generic_csv import GenericCSVDataset

from network import MODEL_REGISTRY, DEFAULT_EXPERTS
from network.moe import MoEDiffuser


from training.runtime import (
    _ProgressReporter,
    _append_jsonl,
    _bcl_to_blc,
    _bcl_to_lc,
    _compute_moe_loss_components,
    _compute_target_recon_loss,
    _distribution_entropy,
    _evaluate_subset_metrics,
    _expert_smoke_test,
    _export_channel_best_expert,
    _format_duration,
    _get_target_names,
    _jsonable_metrics,
    _limit_eval_indices,
    _optimizer_params,
    _progress_bar,
    _resolve_expert_health_signal,
    _resolve_horizon_boundaries,
    _resolve_target_loss_weights,
    _safe_tag,
    _stringify_time_value,
    _temporary_expert_gate_scale,
    _to_numpy,
    _write_csv_rows,
    build_dataset,
    collect_moe_expert_health,
    export_moe_logs,
    forward_diffusion_model,
    get_model_instance,
    is_amp_enabled,
    log_moe_status,
    make_autocast_context,
    make_cpu_generator,
    make_grad_scaler,
    maybe_update_expert_lifecycle,
    optimizer_backward_step,
    resolve_grad_clip_norm,
    run_evaluation,
    select_eval_indices,
    set_global_seed,
    train_moe_stacking,
    train_standard,
    unpack_batch,
    unpack_sample,
)


def _safe_run_tag(value) -> str:
    tag = str(value or "").strip()
    if not tag:
        return ""
    safe = []
    for ch in tag:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def _auto_select_csv(data_dir: str = "csv_data") -> str:
    """Pick the first CSV found in *data_dir*."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[Data] '{data_dir}' directory not found.")
    csv_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".csv") and not f.startswith("~")
    )
    if not csv_files:
        raise FileNotFoundError(f"[Data] No .csv files found in '{data_dir}/'.")
    path = os.path.join(data_dir, csv_files[0])
    print(f"[Data] Auto-selected: {path}")
    return path


def _load_config_file(config_path: str) -> dict:
    original_path = str(config_path)
    resolved_path = os.path.abspath(os.path.expanduser(original_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            "[Config] --config was provided but the file does not exist: "
            f"{original_path} (resolved: {resolved_path}). "
            "On Linux/macOS, use forward slashes, e.g. configs/moe_hierarchical_v1.json."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"[Config] config file must contain a JSON object: {resolved_path}")

    print(f"[Config] loaded from: {resolved_path}")
    return loaded


def main():
    config = {
        "seq_len": 168,
        "context_len": 168,
        "pred_len": 168,
        "batch_size": 32,
        "epochs": 50,
        "phase1_epochs": 20,
        "phase2_epochs": 30,
        "eval_every": 5,
        "val_metric_num_samples": 128,
        "lr": 1e-3,
        "lambda_balance": 1e-3,
        "lambda_entropy": 5e-4,
        "lambda_min_usage": 5e-3,
        "lambda_channel_prior": 1e-3,
        "min_expert_usage": 0.02,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "deterministic": True,
        "use_amp": True,
        "grad_clip_norm": 1.0,
        "csv_path": None,
        "model": "MoEDiffuser",
        "run_tag": "",
        "moe_experts": list(DEFAULT_EXPERTS),
        "label_dir": "labels",
        "add_time_features": True,
        "add_cyc_features": True,
        "eval_sampler": "ddim",
        "eval_inference_steps": 100,
        "eval_eta": 0.0,
        "drift_correction": True,
        "drift_tail_window": 24,
        "drift_alpha": 0.75,
        "use_rule_coarse": True,
        "routing_mode": "hard_channel",
        "expert_diffusion_profile_policy": "require_global_alignment",
        "expert_reference_root": None,
        "coarse_mix_alpha": 0.7,
        "rule_table": None,
        "use_channel_expert_prior": True,
        "channel_prior_alpha": 0.35,
        "channel_expert_prior": None,
        "channel_gate_use_stats": True,
        "channel_gate_stat_hidden_dim": 32,
        "channel_gate_use_embedding": True,
        "use_task_heads": True,
        "task_head_hidden_dim": 16,
        "coarse_top_k": 3,
        "final_top_k": 1,
        "hard_final_routing": True,
        "hard_routing_straight_through": True,
        "gate_temperature": 1.25,
        "soft_retire_scale": 0.1,
        "min_alive_experts": 3,
        "enable_expert_lifecycle": True,
        "protect_channel_specialists": True,
        "channel_specialist_selected_threshold": 0.5,
        "channel_specialist_contribution_threshold": 0.0,
        "phase0_rule_t_values": 5,
        "expert_retire_start_epoch": 5,
        "expert_retire_check_every": 1,
        "soft_retire_patience": 2,
        "hard_prune_patience": 4,
        "expert_health_max_windows": 128,
        "soft_retire_contribution_threshold": 0.0,
        "hard_prune_contribution_threshold": 0.0,
        "soft_retire_usage_threshold": 0.02,
        "hard_prune_usage_threshold": 0.005,
        "horizon_boundaries": None,
        "final_eval_export_all_windows": False,
        "final_eval_num_samples": 128,
        "final_eval_save_window_plots": False,
        "final_eval_window_plot_limit": 0,
        "final_eval_save_overview_plots": True,
        "final_eval_sampler": "ddim",
        "final_eval_inference_steps": 100,
        "final_eval_eta": 0.0,
        "predict_sampler": "ddim",
        "predict_inference_steps": 100,
        "predict_eta": 0.0,
        "target_loss_weights": None,
        "target_only_diffusion": True,
        "forecast_objective": "direct",
        "seasonal_period": 24,
        "baseline_reference": "daily_seasonal",
        "ddim_t_start": 800,
        "eval_batch_size": 1,
        "strict_sample_shape": True,
        "require_ddim_t_start": True,
        "require_target_only_with_time_features": True,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a single config key (repeatable). Values are parsed as JSON when possible.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    if args.config:
        config.update(_load_config_file(args.config))
    if args.csv_path:
        config["csv_path"] = args.csv_path
    if args.model:
        config["model"] = args.model
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.run_tag is not None:
        config["run_tag"] = args.run_tag
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects a non-empty key, got {item!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        config[key] = value
        print(f"[Config] override {key} = {value!r}")

    validate_config(config, mode="train")
    context_len, pred_len = resolve_context_pred_lengths(config)

    if str(config.get("device", "cpu")).lower() == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA unavailable, falling back to CPU.")
        config["device"] = "cpu"

    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    split_kwargs = {
        "split_borders": config.get("split_borders", None),
        "split_ratios": config.get("split_ratios", None),
        "scaler_type": str(config.get("scaler_type", "minmax")),
    }

    # --- Data source ---
    if config["csv_path"] is None:
        config["csv_path"] = _auto_select_csv("csv_data")

    csv_path = config["csv_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Data] csv_path does not exist: {csv_path}")
    if not csv_path.lower().endswith(".csv"):
        raise ValueError(f"[Data] csv_path is not a .csv file: {csv_path}")

    # --- Model ---
    if not config.get("model"):
        config["model"] = "MoEDiffuser"

    if config["model"] == "MoEDiffuser" and not config.get("moe_experts"):
        config["moe_experts"] = list(DEFAULT_EXPERTS)

    # --- Summary ---
    model_name = config["model"]
    dataset_name = os.path.basename(csv_path).split(".")[0]
    num_experts = len(config.get("moe_experts", DEFAULT_EXPERTS))
    print(f"\n[Config] data={os.path.basename(csv_path)}  model={model_name}  "
          f"experts={num_experts}  device={config['device']}")

    # --- Setup ---
    set_global_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)))

    DEVICE = torch.device(config["device"])
    CONTEXT_LEN = int(config["context_len"])
    PRED_LEN = int(config["pred_len"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _safe_run_tag(config.get("run_tag", ""))
    tag_part = f"_{run_tag}" if run_tag else ""
    run_name = f"{model_name}_{dataset_name}{tag_part}_{timestamp}"
    results_root = str(config.get("results_dir", "results"))
    experiment_dir = os.path.join(results_root, run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    log_path = setup_run_log(experiment_dir, "train.txt")
    checkpoint_log_path = os.path.join(experiment_dir, "log", "checkpoint.txt")
    os.makedirs(os.path.dirname(checkpoint_log_path), exist_ok=True)
    open(checkpoint_log_path, "a", encoding="utf-8").close()
    print(f"[Log] terminal  -> {log_path}")
    print(f"[Log] checkpoint -> {checkpoint_log_path}")

    with open(os.path.join(experiment_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # --- Datasets ---
    train_label_path = resolve_label_path(config, csv_path, "train")
    val_label_path = resolve_label_path(config, csv_path, "val")

    print("\n>>> Loading Data...")
    full_train_ds = build_dataset(
        GenericCSVDataset,
        seq_len=PRED_LEN,
        csv_path=csv_path,
        mode="train",
        **split_kwargs,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        channels_first=True,
        add_time_features=bool(config.get("add_time_features", False)),
        add_cyc_features=bool(config.get("add_cyc_features", False)),
        label_path=train_label_path,
        target_only_diffusion=bool(config.get("target_only_diffusion", False)),
    )
    val_ds = build_dataset(
        GenericCSVDataset,
        seq_len=PRED_LEN,
        csv_path=csv_path,
        mode="val",
        **split_kwargs,
        train_ds=full_train_ds,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        channels_first=True,
        add_time_features=bool(config.get("add_time_features", False)),
        add_cyc_features=bool(config.get("add_cyc_features", False)),
        label_path=val_label_path,
        target_only_diffusion=bool(config.get("target_only_diffusion", False)),
    )

    test_ds = None
    if bool(config.get("eval_on_test", True)):
        test_label_path = resolve_label_path(config, csv_path, "test")
        test_ds = build_dataset(
            GenericCSVDataset,
            seq_len=PRED_LEN,
            csv_path=csv_path,
            mode="test",
        **split_kwargs,
            train_ds=full_train_ds,
            context_len=CONTEXT_LEN,
            pred_len=PRED_LEN,
            channels_first=True,
            add_time_features=bool(config.get("add_time_features", False)),
            add_cyc_features=bool(config.get("add_cyc_features", False)),
            label_path=test_label_path,
            target_only_diffusion=bool(config.get("target_only_diffusion", False)),
        )
        if len(test_ds) == 0:
            print("[Data] test split is empty -> held-out test evaluation disabled")
            test_ds = None
        else:
            print(f"[Data] held-out test windows={len(test_ds)}")

    NUM_CHANNELS = full_train_ds.num_channels
    first_sample = full_train_ds[0]
    sample_shape = first_sample["past"].shape if isinstance(first_sample, dict) else first_sample[0].shape
    print(f"[Data] context={CONTEXT_LEN}  pred={PRED_LEN}  channels={NUM_CHANNELS}  "
          f"sample_shape={tuple(sample_shape)}")

    if val_ds.num_channels != NUM_CHANNELS:
        raise ValueError(f"[Data] Channel mismatch: train={NUM_CHANNELS}, val={val_ds.num_channels}")

    # --- Save dataset assets ---
    if hasattr(full_train_ds, "scaler"):
        joblib.dump(full_train_ds.scaler, os.path.join(experiment_dir, "scaler.pkl"))
    if hasattr(full_train_ds, "feature_cols"):
        with open(os.path.join(experiment_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
            json.dump(full_train_ds.feature_cols, f, indent=2, ensure_ascii=False)
    if hasattr(full_train_ds, "target_cols"):
        with open(os.path.join(experiment_dir, "target_cols.json"), "w", encoding="utf-8") as f:
            json.dump(full_train_ds.target_cols, f, indent=2, ensure_ascii=False)
    if hasattr(full_train_ds, "known_feature_cols"):
        with open(os.path.join(experiment_dir, "known_feature_cols.json"), "w", encoding="utf-8") as f:
            json.dump(full_train_ds.known_feature_cols, f, indent=2, ensure_ascii=False)

    # --- Build model ---
    target_indices = getattr(full_train_ds, "target_indices", None)
    known_future_channels = int(getattr(full_train_ds, "known_future_channels", 0))
    model = get_model_instance(
        config, NUM_CHANNELS, DEVICE,
        target_indices=target_indices, known_future_channels=known_future_channels,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                   weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = DDPMScheduler(
        num_timesteps=1000,
        device=DEVICE,
        schedule=config.get("noise_schedule", "linear"),
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )

    # --- Train ---
    if model_name == "MoEDiffuser":
        print("\n>>> [Mode: MoE Stacking] Phase 1 (experts + gate) -> Phase 2 (gate only)")
        train_moe_stacking(model, full_train_ds, val_ds, optimizer, scheduler, config, experiment_dir, DEVICE, test_ds=test_ds)
    else:
        print(f"\n>>> [Mode: Standard] {model_name}")
        train_standard(model, full_train_ds, val_ds, optimizer, scheduler, config, experiment_dir, DEVICE)

    print(f"\n[Done] {experiment_dir}")


if __name__ == "__main__":
    main()
