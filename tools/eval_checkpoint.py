import os
import sys
import json
import argparse

# Match the training entrypoint before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.generic_csv import GenericCSVDataset
from utils.run_logging import setup_run_log
from utils.config import validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder", type=str, required=True)
    parser.add_argument("--epoch_name", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_expert_health", action="store_true")
    parser.add_argument("--ensemble_samples", type=int, default=None)
    args = parser.parse_args()
    if args.epoch_name is None:
        args.epoch_name = "FinalFromCheckpoint" if args.split == "val" else "FinalTestFromCheckpoint"

    experiment_folder = os.path.abspath(args.experiment_folder)
    log_path = setup_run_log(experiment_folder, "checkpoint.txt")
    print(f"[Log] Checkpoint evaluation terminal output -> {log_path}")

    from train import (
        DDPMScheduler,
        _export_channel_best_expert,
        _get_target_names,
        _safe_tag,
        build_dataset,
        export_moe_logs,
        get_model_instance,
        resolve_context_pred_lengths,
        resolve_label_path,
        run_evaluation,
        set_global_seed,
        _resolve_horizon_boundaries,
    )

    config_path = os.path.join(experiment_folder, "config.json")
    weights_path = os.path.join(experiment_folder, "model_weights.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config, mode="eval")
    context_len, pred_len = resolve_context_pred_lengths(config)
    config["context_len"] = int(context_len)
    config["pred_len"] = int(pred_len)
    config["seq_len"] = int(pred_len)

    if str(config.get("device", "cpu")).lower() == "cuda" and not torch.cuda.is_available():
        print("[EvalOnly] CUDA unavailable, falling back to CPU.")
        config["device"] = "cpu"

    set_global_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)))

    device = torch.device(config["device"])
    csv_path = config.get("csv_path", None)
    if not csv_path:
        raise RuntimeError("config.json is missing csv_path")

    split_kwargs = {
        "split_borders": config.get("split_borders", None),
        "split_ratios": config.get("split_ratios", None),
        "scaler_type": str(config.get("scaler_type", "minmax")),
    }

    train_label_path = resolve_label_path(config, csv_path, "train")
    val_label_path = resolve_label_path(config, csv_path, "val")

    full_train_ds = build_dataset(
        GenericCSVDataset,
        seq_len=pred_len,
        csv_path=csv_path,
        mode="train",
        **split_kwargs,
        context_len=context_len,
        pred_len=pred_len,
        channels_first=True,
        add_time_features=bool(config.get("add_time_features", False)),
        add_cyc_features=bool(config.get("add_cyc_features", False)),
        label_path=train_label_path,
        target_only_diffusion=bool(config.get("target_only_diffusion", False)),
    )
    val_ds = build_dataset(
        GenericCSVDataset,
        seq_len=pred_len,
        csv_path=csv_path,
        mode="val",
        **split_kwargs,
        train_ds=full_train_ds,
        context_len=context_len,
        pred_len=pred_len,
        channels_first=True,
        add_time_features=bool(config.get("add_time_features", False)),
        add_cyc_features=bool(config.get("add_cyc_features", False)),
        label_path=val_label_path,
        target_only_diffusion=bool(config.get("target_only_diffusion", False)),
    )

    eval_ds = val_ds
    if args.split == "test":
        test_label_path = resolve_label_path(config, csv_path, "test")
        print("[EvalOnly] building held-out test split")
        eval_ds = build_dataset(
            GenericCSVDataset,
            seq_len=pred_len,
            csv_path=csv_path,
            mode="test",
        **split_kwargs,
            train_ds=full_train_ds,
            context_len=context_len,
            pred_len=pred_len,
            channels_first=True,
            add_time_features=bool(config.get("add_time_features", False)),
            add_cyc_features=bool(config.get("add_cyc_features", False)),
            label_path=test_label_path,
            target_only_diffusion=bool(config.get("target_only_diffusion", False)),
        )
        if len(eval_ds) == 0:
            raise RuntimeError("test split is empty")

    target_indices = getattr(full_train_ds, "target_indices", None)
    model = get_model_instance(
        config,
        int(full_train_ds.num_channels),
        device,
        target_indices=target_indices,
        known_future_channels=int(getattr(full_train_ds, "known_future_channels", 0)),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    scheduler = DDPMScheduler(
        num_timesteps=1000,
        device=device,
        schedule=config.get("noise_schedule", "linear"),
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )

    seed = int(config.get("seed", 42))
    index_seed = seed + 1001
    sample_seed = seed + 2001
    final_eval_sampler = str(config.get("final_eval_sampler", config.get("eval_sampler", "ddim")))
    final_eval_inference_steps = int(
        config.get("final_eval_inference_steps", config.get("eval_inference_steps", 100))
    )
    final_eval_eta = float(config.get("final_eval_eta", config.get("eval_eta", 0.0)))
    horizon_boundaries = _resolve_horizon_boundaries(config)

    export_all_windows = args.max_eval_windows is None
    max_eval_windows = None if export_all_windows else int(args.max_eval_windows)

    is_hard_channel = str(config.get("routing_mode", "")).lower() == "hard_channel"
    save_expert_health = bool(args.save_expert_health) or is_hard_channel
    final_metrics = run_evaluation(
        model,
        scheduler,
        eval_ds,
        args.epoch_name,
        device,
        experiment_folder,
        export_all_windows=export_all_windows,
        max_eval_windows=max_eval_windows,
        index_seed=index_seed,
        sample_seed=sample_seed,
        save_window_plots=bool(config.get("final_eval_save_window_plots", False)),
        window_plot_limit=config.get("final_eval_window_plot_limit", 0),
        save_overview_plots=bool(config.get("final_eval_save_overview_plots", True)),
        sample_sampler=final_eval_sampler,
        sample_inference_steps=final_eval_inference_steps,
        sample_eta=final_eval_eta,
        sample_t_start=config.get("ddim_t_start", None),
        use_amp=bool(config.get("use_amp", True)) and device.type == "cuda",
        drift_correction=bool(config.get("drift_correction", True)),
        drift_tail_window=int(config.get("drift_tail_window", 24)),
        drift_alpha=float(config.get("drift_alpha", 0.75)),
        horizon_boundaries=horizon_boundaries,
        save_expert_health=save_expert_health,
        expert_health_max_windows=int(config.get("expert_health_max_windows", 32)),
        crps_samples=int(config.get("crps_samples", 0)),
        crps_num_windows=int(config.get("crps_num_windows", 16)),
        eval_batch_size=int(args.eval_batch_size or config.get("eval_batch_size", 1)),
        strict_sample_shape=bool(config.get("strict_sample_shape", True)),
        forecast_objective=str(config.get("forecast_objective", "direct")),
        seasonal_period=int(config.get("seasonal_period", 24)),
        baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
        eval_space=str(config.get("eval_space", "raw")),
        eval_ensemble_samples=int(args.ensemble_samples or config.get("eval_ensemble_samples", 1)),
        inference_mode=str(config.get("inference_mode", "diffusion")),
    )
    if is_hard_channel:
        target_idx = getattr(eval_ds, "target_indices", None) or list(range(getattr(eval_ds, "num_channels", 1)))
        target_names = _get_target_names(eval_ds, target_idx)
        payload = _export_channel_best_expert(
            experiment_folder,
            target_names,
            getattr(model, "expert_names", []),
            final_metrics,
            file_stem=f"channel_best_expert_{_safe_tag(args.epoch_name)}",
        )
        if payload is not None:
            print(f"[EvalOnly] Saved channel_best_expert_{_safe_tag(args.epoch_name)}.json/.csv")
        export_moe_logs(model, eval_ds, device, experiment_folder)


if __name__ == "__main__":
    main()
