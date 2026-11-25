import json
import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from utils.metrics import resolve_horizon_ranges
from training.core import forward_diffusion_model, unpack_batch, unpack_sample
from training.common import (
    _compute_target_recon_loss,
    _get_target_names,
    _resolve_horizon_boundaries,
    _resolve_target_loss_weights,
    _write_csv_rows,
    is_amp_enabled,
    make_autocast_context,
    make_cpu_generator,
    make_grad_scaler,
    optimizer_backward_step,
    resolve_grad_clip_norm,
    select_eval_indices,
)
from training.evaluation import run_evaluation
from training.health import (
    _compute_moe_loss_components,
    _evaluate_subset_metrics,
    maybe_update_expert_lifecycle,
)
from training.baselines import (
    make_residual_target_bcl,
    normalize_forecast_objective,
    uses_residual_forecast_objective,
)


EXPERT_REFERENCE_RELATIVE_PATHS = {
    "UNetDiffuser": "DDPM-UNet",
    "DiffWaveDiffuser": "DiffWave-1DCNN",
    "PatchTransformerDiffuser": "Patch-MLP",
    "UGNetDiffuser": "DiffSTG-GNN",
    "CSDIAttentionDiffuser": "CSDI-Transformer",
    "SpectralDiffuser": "CSDI-Transformer",
    "SelectiveSSMDiffuser": "SSSD-S4 Mamba SSM",
    "DiMTSMambaDiffuser": "SSSD-S4 Mamba SSM",
    "RNNDiffuser": "RNN",
}


def _resolve_forecast_objective(config):
    objective = normalize_forecast_objective(config.get("forecast_objective", "direct"))
    period = int(max(1, config.get("seasonal_period", 24)))
    return objective, period


def _make_diffusion_target(past, future, forecast_objective: str, seasonal_period: int):
    if uses_residual_forecast_objective(forecast_objective):
        return make_residual_target_bcl(past, future, period=seasonal_period)
    return future


def _limit_training_dataset(dataset, config, *, seed_offset: int = 0):
    # `max_train_windows` is a compute-budget control (not a debug switch): it
    # keeps the number of optimizer steps per epoch comparable across datasets.
    max_windows = config.get("debug_max_train_windows", None)
    if max_windows is None:
        max_windows = config.get("max_train_windows", None)
    if max_windows is None:
        return dataset

    max_windows = int(max_windows)
    if max_windows <= 0 or max_windows >= len(dataset):
        return dataset

    seed = int(config.get("seed", 42)) + int(seed_offset)
    indices = select_eval_indices(len(dataset), max_windows, seed=seed)
    print(f"[Data] training windows limited to {len(indices)}/{len(dataset)} (max_train_windows)")
    return Subset(dataset, indices)


def _read_csv_rows(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _export_channel_best_expert(save_dir, target_names, expert_names, final_metrics, file_stem="channel_best_expert"):
    summary_rows = final_metrics.get("channel_expert_summary_rows") if final_metrics else None
    if not summary_rows:
        return None

    prior_rows = _read_csv_rows(os.path.join(save_dir, "channel_expert_prior.csv"))
    train_loss_rows = _read_csv_rows(os.path.join(save_dir, "channel_train_loss.csv"))

    prior_best = {}
    for row in prior_rows:
        # Newer exports carry an explicit channel_pos; older ones only had
        # channel_idx (which used to hold the position, not the column index).
        channel_pos = int(row.get("channel_pos", row.get("channel_idx", 0)))
        current = prior_best.get(channel_pos)
        weight = float(row.get("prior_weight", 0.0))
        if current is None or weight > float(current.get("prior_weight", 0.0)):
            prior_best[channel_pos] = row

    latest_train_loss = {}
    for row in train_loss_rows:
        key = (str(row.get("phase", "")), int(row.get("channel_pos", 0)))
        current = latest_train_loss.get(key)
        epoch = int(row.get("epoch", 0))
        if current is None or epoch >= int(current.get("epoch", 0)):
            latest_train_loss[key] = row

    best_rows = []
    best_payload = {
        "target_names": list(target_names),
        "expert_names": list(expert_names),
        "channels": [],
    }
    for row in summary_rows:
        channel_pos = int(row["channel_pos"])
        channel_name = str(row["channel_name"])
        prior_row = prior_best.get(channel_pos, {})
        p1_row = latest_train_loss.get(("phase1", channel_pos), {})
        p2_row = latest_train_loss.get(("phase2", channel_pos), {})
        best_row = {
            "channel_idx": int(row["channel_idx"]),
            "channel_pos": channel_pos,
            "channel_name": channel_name,
            "prior_expert_name": str(prior_row.get("expert_name", "")),
            "prior_weight": float(prior_row.get("prior_weight", 0.0) or 0.0),
            "prior_avg_mse_loss": float(prior_row.get("avg_mse_loss", 0.0) or 0.0),
            "top_selected_expert_name": str(row["top_selected_expert_name"]),
            "top_selected_freq": float(row["top_selected_freq"]),
            "top_dense_expert_name": str(row["top_dense_expert_name"]),
            "top_dense_weight": float(row["top_dense_weight"]),
            "top_contribution_expert_name": str(row["top_contribution_expert_name"]),
            "top_contribution_score": float(row["top_contribution_score"]),
            "phase1_recon_mse": float(p1_row.get("recon_mse", 0.0) or 0.0),
            "phase2_recon_mse": float(p2_row.get("recon_mse", 0.0) or 0.0),
        }
        best_rows.append(best_row)
        best_payload["channels"].append(best_row)

    _write_csv_rows(
        os.path.join(save_dir, f"{file_stem}.csv"),
        best_rows,
        [
            "channel_idx",
            "channel_pos",
            "channel_name",
            "prior_expert_name",
            "prior_weight",
            "prior_avg_mse_loss",
            "top_selected_expert_name",
            "top_selected_freq",
            "top_dense_expert_name",
            "top_dense_weight",
            "top_contribution_expert_name",
            "top_contribution_score",
            "phase1_recon_mse",
            "phase2_recon_mse",
        ],
    )
    with open(os.path.join(save_dir, f"{file_stem}.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)
    return best_payload


def _default_expert_reference_root() -> str:
    """Reference implementations live outside the repo; keep the path portable."""
    override = os.environ.get("MOE_EXPERT_REFERENCE_ROOT", "").strip()
    return override or "MMoediffusionmodels"


def _export_expert_diffusion_profiles(model, config, save_dir):
    if not hasattr(model, "expert_diffusion_profiles"):
        return None
    reference_root = str(
        config.get("expert_reference_root") or _default_expert_reference_root()
    )
    prediction_type = str(config.get("prediction_type", "epsilon")).strip().lower()
    training_mode = str(config.get("training_mode", "diffusion")).strip().lower()
    objective_name = "noise_prediction" if prediction_type == "epsilon" else f"{prediction_type}_prediction"
    if training_mode == "regression":
        objective_name = "direct_regression"
    global_diffusion = {
        "noise_schedule": str(config.get("noise_schedule", "linear")),
        "num_timesteps": 1000,
        "training_objective": objective_name,
        "training_timestep_sampling": "uniform_discrete",
        "eval_sampler": str(config.get("eval_sampler", "ddim")),
        "eval_inference_steps": int(config.get("eval_inference_steps", 100)),
        "final_eval_sampler": str(config.get("final_eval_sampler", config.get("eval_sampler", "ddim"))),
        "final_eval_inference_steps": int(
            config.get("final_eval_inference_steps", config.get("eval_inference_steps", 100))
        ),
        "ddim_t_start": config.get("ddim_t_start", None),
        "eval_eta": float(config.get("eval_eta", 0.0)),
        "final_eval_eta": float(config.get("final_eval_eta", config.get("eval_eta", 0.0))),
        "profile_policy": str(config.get("expert_diffusion_profile_policy", "require_global_alignment")),
        "reference_model_root": reference_root,
    }
    architecture_state = {
        "routing_granularity": "target_channel",
        "routing_mode": str(config.get("routing_mode", "")),
        "routing_optimization_status": "deferred",
        "expert_selection": "hard_one_expert_per_channel",
        "current_expert_integration": "denoiser_extracted_under_shared_scheduler",
        "target_expert_integration": "full_diffusion_expert_per_channel",
    }
    profiles = []
    for raw_profile in model.expert_diffusion_profiles():
        profile = dict(raw_profile)
        expert_name = str(profile.get("expert_name", profile.get("class_name", "")))
        rel_ref = EXPERT_REFERENCE_RELATIVE_PATHS.get(expert_name, "")
        if rel_ref and not profile.get("source_reference_path"):
            profile["source_reference_path"] = str(Path(reference_root) / rel_ref)
        profile["current_integration_level"] = str(profile.get("current_integration_level", "denoiser_extracted"))
        profile["target_integration_level"] = str(profile.get("target_integration_level", "full_diffusion_expert"))
        profile["training_objective"] = global_diffusion["training_objective"]
        profile["training_timestep_sampling"] = global_diffusion["training_timestep_sampling"]
        profile["scheduler_scope"] = "shared_global"
        profile["noise_schedule"] = global_diffusion["noise_schedule"]
        profile["num_timesteps"] = global_diffusion["num_timesteps"]
        sampling = dict(profile.get("sampling") or {})
        sampling.update(
            {
                "sampler": global_diffusion["final_eval_sampler"],
                "ddim_t_start": global_diffusion["ddim_t_start"],
                "eta": global_diffusion["final_eval_eta"],
                "inference_steps": global_diffusion["final_eval_inference_steps"],
            }
        )
        profile["sampling"] = sampling
        profile["shared_scheduler_aligned"] = True
        profile["effective_diffusion_protocol"] = "shared_global_scheduler_expert_denoiser"
        required = [str(item) for item in profile.get("required_full_protocol_operations", [])]
        full_protocol = set(str(item) for item in profile.get("full_protocol_operations", []))
        preserved = set(str(item) for item in profile.get("preserved_operations", []))
        profile["missing_full_protocol_operations"] = [
            item for item in required if item not in full_protocol and item not in preserved
        ]
        profiles.append(profile)
    payload = {
        "architecture_state": architecture_state,
        "global_diffusion": global_diffusion,
        "profiles": profiles,
    }
    with open(os.path.join(save_dir, "expert_diffusion_profiles.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def log_moe_status(
    model,
    past_batch,
    epoch,
    prefix="",
    coarse_label_batch=None,
    known_future_batch=None,
    task_names=None,
):
    model.eval()
    with torch.no_grad():
        _, weight_dict = forward_diffusion_model(
            model,
            None,
            None,
            past_batch,
            coarse_label=coarse_label_batch,
            known_future=known_future_batch,
            return_weights=True,
        )

        final_avg = weight_dict["final"].mean(dim=(0, 1)).cpu().numpy()
        coarse = weight_dict["coarse"]
        if coarse.dim() == 2:
            coarse_avg = coarse.mean(dim=0).cpu().numpy()
        else:
            coarse_avg = coarse.mean(dim=(0, 1)).cpu().numpy()
        fine_avg = weight_dict["fine"].mean(dim=(0, 1)).cpu().numpy()
        task_final_avg = weight_dict["final"].mean(dim=0).cpu().numpy()

    names = model.expert_names
    coarse_info = " | ".join([f"{n[:8]}:{w:.3f}" for n, w in zip(names, coarse_avg)])
    fine_info = " | ".join([f"{n[:8]}:{w:.3f}" for n, w in zip(names, fine_avg)])
    hard_routing = bool(getattr(model.mmoe_gating, "hard_final_routing", False))
    final_label = "SelectFreq" if hard_routing else "Final"
    final_info = " | ".join([f"{n[:8]}:{w:.3f}" for n, w in zip(names, final_avg)])

    print(f"   📊 [{prefix} Ep{epoch}] Coarse: {coarse_info}")
    print(f"   📊 [{prefix} Ep{epoch}] Fine  : {fine_info}")
    print(f"   📊 [{prefix} Ep{epoch}] {final_label}: {final_info}")
    if task_names is None:
        task_names = [f"ch{idx}" for idx in getattr(model, "target_indices", range(task_final_avg.shape[0]))]
    top_k = min(3, len(names))
    for task_i, task_weights in enumerate(task_final_avg):
        order = np.argsort(task_weights)[::-1][:top_k]
        task_name = str(task_names[task_i]) if task_i < len(task_names) else f"task{task_i}"
        top_info = " | ".join([f"{names[j][:8]}:{task_weights[j]:.3f}" for j in order])
        route_label = "TaskSelect" if hard_routing else "TaskRoute"
        print(f"   [{route_label} {prefix} Ep{epoch}] {task_name}: {top_info}")
    model.train()


def train_standard(model, train_ds, val_ds, optimizer, scheduler, config, save_dir, device):
    train_loader_ds = _limit_training_dataset(train_ds, config, seed_offset=31)
    train_loader = DataLoader(
        train_loader_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=make_cpu_generator(int(config.get("seed", 42)) + 11),
    )
    model.train()

    target_idx = getattr(train_ds, "target_indices", None)
    if not target_idx:
        target_idx = list(range(getattr(train_ds, "num_channels", 1)))
    target_names = _get_target_names(train_ds, target_idx)
    target_loss_weights = _resolve_target_loss_weights(config, target_names)
    if target_loss_weights is not None:
        weights_text = " | ".join(
            f"{name}:{float(weight):.2f}" for name, weight in zip(target_names, target_loss_weights.tolist())
        )
        print(f"[Loss] target_loss_weights -> {weights_text}")
    forecast_objective, seasonal_period = _resolve_forecast_objective(config)
    prediction_type = str(config.get("prediction_type", "epsilon")).strip().lower()
    if uses_residual_forecast_objective(forecast_objective):
        print(f"[Objective] forecast_objective={forecast_objective} | seasonal_period={seasonal_period}")
    print(f"[Objective] prediction_type={prediction_type}")

    use_amp = is_amp_enabled(config, device)
    grad_clip_norm = resolve_grad_clip_norm(config)
    scaler = make_grad_scaler(use_amp)
    eval_every = max(1, int(config.get("eval_every", 5)))
    val_metric_num_samples = int(config.get("val_metric_num_samples", 32))
    eval_index_seed = int(config.get("seed", 42)) + 1001
    eval_sample_seed = int(config.get("seed", 42)) + 2001
    eval_sampler = str(config.get("eval_sampler", "ddim"))
    eval_inference_steps = int(config.get("eval_inference_steps", 100))
    eval_eta = float(config.get("eval_eta", 0.0))
    eval_batch_size = int(config.get("eval_batch_size", 1))
    strict_sample_shape = bool(config.get("strict_sample_shape", True))
    final_eval_sampler = str(config.get("final_eval_sampler", eval_sampler))
    final_eval_inference_steps = int(config.get("final_eval_inference_steps", eval_inference_steps))
    final_eval_eta = float(config.get("final_eval_eta", eval_eta))
    final_eval_export_all_windows = bool(config.get("final_eval_export_all_windows", True))
    final_eval_num_samples = None
    if not final_eval_export_all_windows:
        final_eval_num_samples = int(config.get("final_eval_num_samples", val_metric_num_samples))
    horizon_boundaries = _resolve_horizon_boundaries(config)
    channel_train_loss_rows = []
    best_rmse = float("inf")
    best_epoch = None

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for batch in train_loader:
            past, future, _, known_future = unpack_batch(batch, device)

            t = torch.randint(0, 1000, (past.shape[0],), device=device).long()
            diffusion_target = _make_diffusion_target(past, future, forecast_objective, seasonal_period)
            noise = torch.randn_like(diffusion_target)
            noisy_future = scheduler.add_noise(diffusion_target, noise, t)

            with make_autocast_context(device, use_amp):
                pred_noise = forward_diffusion_model(
                    model,
                    noisy_future,
                    t,
                    past,
                    known_future=known_future,
                )
                recon_reference = noise if prediction_type == "epsilon" else diffusion_target
                loss = _compute_target_recon_loss(
                    pred_noise,
                    recon_reference,
                    target_idx,
                    target_loss_weights=target_loss_weights,
                )

            optimizer_backward_step(loss, optimizer, scaler, grad_clip_norm=grad_clip_norm)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.6f}")

        should_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == config["epochs"])
        if should_eval:
            metrics = run_evaluation(
                model,
                scheduler,
                val_ds,
                f"Ep{epoch+1}",
                device,
                save_dir,
                export_all_windows=False,
                max_eval_windows=val_metric_num_samples,
                index_seed=eval_index_seed,
                sample_seed=eval_sample_seed,
                save_preview_plot=True,
                save_window_plots=False,
                save_overview_plots=False,
                sample_sampler=eval_sampler,
                sample_inference_steps=eval_inference_steps,
                sample_eta=eval_eta,
                sample_t_start=config.get("ddim_t_start", None),
                use_amp=use_amp,
                drift_correction=bool(config.get("drift_correction", True)),
                drift_tail_window=int(config.get("drift_tail_window", 24)),
                drift_alpha=float(config.get("drift_alpha", 0.75)),
                horizon_boundaries=horizon_boundaries,
                eval_batch_size=eval_batch_size,
                strict_sample_shape=strict_sample_shape,
                forecast_objective=forecast_objective,
                seasonal_period=seasonal_period,
                baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
                eval_space=str(config.get("eval_space", "raw")),
                eval_ensemble_samples=int(config.get("eval_ensemble_samples", 1)),
                inference_mode=str(config.get("inference_mode", "diffusion")),
            )
            if metrics is not None and metrics["RMSE"] < best_rmse:
                best_rmse = float(metrics["RMSE"])
                best_epoch = int(epoch + 1)
                torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
                with open(os.path.join(save_dir, "best_val_metric.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "selection_metric": "RMSE",
                            "selection_split": "val_subset",
                            "subset_windows": int(val_metric_num_samples),
                            "sampler": eval_sampler,
                            "inference_steps": int(eval_inference_steps),
                            "epoch": best_epoch,
                            "RMSE": best_rmse,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"   [Best] Epoch {best_epoch} | ValSubset RMSE={best_rmse:.4f}")

    if best_epoch is None:
        best_epoch = int(config["epochs"])
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    model.load_state_dict(torch.load(os.path.join(save_dir, "model_weights.pth"), map_location=device))
    final_metrics = run_evaluation(
        model,
        scheduler,
        val_ds,
        "Final",
        device,
        save_dir,
        export_all_windows=final_eval_export_all_windows,
        max_eval_windows=final_eval_num_samples,
        index_seed=eval_index_seed,
        sample_seed=eval_sample_seed,
        save_window_plots=bool(config.get("final_eval_save_window_plots", False)),
        window_plot_limit=config.get("final_eval_window_plot_limit", 0),
        save_overview_plots=bool(config.get("final_eval_save_overview_plots", True)),
        sample_sampler=final_eval_sampler,
        sample_inference_steps=final_eval_inference_steps,
        sample_eta=final_eval_eta,
        sample_t_start=config.get("ddim_t_start", None),
        use_amp=use_amp,
        drift_correction=bool(config.get("drift_correction", True)),
        drift_tail_window=int(config.get("drift_tail_window", 24)),
        drift_alpha=float(config.get("drift_alpha", 0.75)),
        horizon_boundaries=horizon_boundaries,
        eval_batch_size=eval_batch_size,
        strict_sample_shape=strict_sample_shape,
        forecast_objective=forecast_objective,
        seasonal_period=seasonal_period,
        baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
        eval_space=str(config.get("eval_space", "raw")),
        eval_ensemble_samples=int(config.get("eval_ensemble_samples", 1)),
        inference_mode=str(config.get("inference_mode", "diffusion")),
    )


def train_moe_stacking(model, full_train_ds, val_ds, optimizer, scheduler, config, save_dir, device, test_ds=None):
    split_source_ds = _limit_training_dataset(full_train_ds, config, seed_offset=41)
    if len(split_source_ds) < 2:
        raise ValueError("[Train] MoE training needs at least 2 windows after debug_max_train_windows limiting.")

    pretrain_epochs = int(config.get("pretrain_epochs", 0))
    len_prior = 0
    if pretrain_epochs > 0 and len(split_source_ds) >= 3:
        len_prior = max(1, int(len(split_source_ds) * float(config.get("prior_calibration_ratio", 0.1))))
        len_prior = min(len_prior, len(split_source_ds) - 2)

    len_remaining = len(split_source_ds) - len_prior
    len_expert = int(len_remaining * 0.8)
    len_expert = max(1, min(len_expert, len_remaining - 1))
    len_gate = len_remaining - len_expert
    split_lengths = [len_expert, len_gate]
    if len_prior > 0:
        split_lengths.append(len_prior)

    split_parts = random_split(
        split_source_ds,
        split_lengths,
        generator=make_cpu_generator(int(config.get("seed", 42)) + 21),
    )
    expert_ds, gate_ds = split_parts[0], split_parts[1]
    prior_calibration_ds = split_parts[2] if len_prior > 0 else gate_ds

    expert_loader = DataLoader(
        expert_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=make_cpu_generator(int(config.get("seed", 42)) + 22),
    )
    gate_loader = DataLoader(
        gate_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=make_cpu_generator(int(config.get("seed", 42)) + 23),
    )

    print(
        f"[Split] expert_train={len(expert_ds)} | "
        f"gate_train={len(gate_ds)} | prior_calibration={len(prior_calibration_ds)}"
    )

    target_idx = getattr(full_train_ds, "target_indices", None)
    if not target_idx:
        target_idx = list(range(getattr(full_train_ds, "num_channels", 1)))
    target_names = getattr(full_train_ds, "target_cols", None)
    if not target_names or len(target_names) != len(target_idx):
        target_names = [f"ch{idx}" for idx in target_idx]
    target_loss_weights = _resolve_target_loss_weights(config, target_names)
    if target_loss_weights is not None:
        weights_text = " | ".join(
            f"{name}:{float(weight):.2f}" for name, weight in zip(target_names, target_loss_weights.tolist())
        )
        print(f"[Loss] target_loss_weights -> {weights_text}")
    forecast_objective, seasonal_period = _resolve_forecast_objective(config)
    prediction_type = str(config.get("prediction_type", "epsilon")).strip().lower()
    training_mode = str(config.get("training_mode", "diffusion")).strip().lower()
    if uses_residual_forecast_objective(forecast_objective):
        print(f"[Objective] forecast_objective={forecast_objective} | seasonal_period={seasonal_period}")
    print(f"[Objective] prediction_type={prediction_type} | training_mode={training_mode}")
    profile_payload = _export_expert_diffusion_profiles(model, config, save_dir)
    if profile_payload is not None:
        print("[Profiles] Saved expert_diffusion_profiles.json")

    use_amp = is_amp_enabled(config, device)
    grad_clip_norm = resolve_grad_clip_norm(config)
    phase1_scaler = make_grad_scaler(use_amp)
    phase2_scaler = make_grad_scaler(use_amp)
    eval_every = max(1, int(config.get("eval_every", 5)))
    val_metric_num_samples = int(config.get("val_metric_num_samples", 32))
    eval_index_seed = int(config.get("seed", 42)) + 1001
    eval_sample_seed = int(config.get("seed", 42)) + 2001
    eval_sampler = str(config.get("eval_sampler", "ddim"))
    eval_inference_steps = int(config.get("eval_inference_steps", 100))
    eval_eta = float(config.get("eval_eta", 0.0))
    eval_batch_size = int(config.get("eval_batch_size", 1))
    strict_sample_shape = bool(config.get("strict_sample_shape", True))
    final_eval_sampler = str(config.get("final_eval_sampler", eval_sampler))
    final_eval_inference_steps = int(config.get("final_eval_inference_steps", eval_inference_steps))
    final_eval_eta = float(config.get("final_eval_eta", eval_eta))
    final_eval_export_all_windows = bool(config.get("final_eval_export_all_windows", True))
    final_eval_num_samples = None
    if not final_eval_export_all_windows:
        final_eval_num_samples = int(config.get("final_eval_num_samples", val_metric_num_samples))
    horizon_boundaries = _resolve_horizon_boundaries(config)
    channel_train_loss_rows = []

    # ── Phase 0: Expert individual pretraining ──────────────
    if pretrain_epochs > 0:
        print(f"\n👉 [Phase 0] 专家单独预训练 ({pretrain_epochs} epochs each)...")
        for p in model.mmoe_gating.parameters():
            p.requires_grad = False
        if getattr(model, "use_task_heads", False):
            # Task heads sit in the output path, so leaving them trainable would
            # let Phase 0 train them on top of each single expert.
            for p in model.task_heads.parameters():
                p.requires_grad = False
        for expert_idx in range(model.num_experts):
            for i in range(model.num_experts):
                if i == expert_idx:
                    model.expert_status[i] = model.STATUS_ACTIVE
                    model.expert_gate_scale[i] = 1.0
                    for p in model.experts[i].parameters(): p.requires_grad = True
                else:
                    model.hard_prune_expert(i, force=True)
            name = model.expert_names[expert_idx]
            print(f"   [{expert_idx+1}/{model.num_experts}] {name}")
            for epoch in range(pretrain_epochs):
                model.train(); epoch_loss = 0.0
                for batch in expert_loader:
                    past, future, coarse_label, known_future = unpack_batch(batch, device)
                    diffusion_target = _make_diffusion_target(past, future, forecast_objective, seasonal_period)
                    if training_mode == "regression":
                        t = torch.zeros(past.shape[0], dtype=torch.long, device=device)
                        noise = torch.zeros_like(diffusion_target)
                        noisy_future = torch.zeros_like(diffusion_target)
                    else:
                        t = torch.randint(0, 1000, (past.shape[0],), device=device).long()
                        noise = torch.randn_like(diffusion_target)
                        noisy_future = scheduler.add_noise(diffusion_target, noise, t)
                    with make_autocast_context(device, use_amp):
                        pred_noise, _ = forward_diffusion_model(
                            model, noisy_future, t, past,
                            coarse_label=coarse_label, known_future=known_future,
                            return_weights=True,
                        )
                        recon_reference = noise if prediction_type == "epsilon" else diffusion_target
                        loss = _compute_target_recon_loss(pred_noise, recon_reference, target_idx,
                            target_loss_weights=target_loss_weights)
                    optimizer_backward_step(loss, optimizer, phase1_scaler, grad_clip_norm=grad_clip_norm)
                    epoch_loss += loss.item()
                if (epoch+1) % 5 == 0 or epoch == 0:
                    print(f"      Epoch {epoch+1:>3}/{pretrain_epochs} | Loss: {epoch_loss/max(1,len(expert_loader)):.6f}")
        for i in range(model.num_experts):
            model.expert_status[i] = model.STATUS_ACTIVE
            model.expert_gate_scale[i] = 1.0
            model.reset_expert_bad_epoch(i)
            for p in model.experts[i].parameters(): p.requires_grad = True
        for p in model.mmoe_gating.parameters():
            p.requires_grad = True
        if getattr(model, "use_task_heads", False):
            for p in model.task_heads.parameters():
                p.requires_grad = True

        # Data-driven rule table from pretrained experts
        print("   [RuleTable] Evaluating pretrained experts per coarse label...")
        prior_max_windows = int(max(1, config.get("phase0_prior_max_windows", 256)))
        prior_indices = select_eval_indices(
            len(prior_calibration_ds),
            min(prior_max_windows, len(prior_calibration_ds)),
            seed=int(config.get("seed", 42)) + 57,
        )
        eval_ds = Subset(prior_calibration_ds, prior_indices)
        model.eval()
        per_loss = torch.zeros(4, model.num_experts, device=device)
        per_cnt = torch.zeros(4, device=device)
        channel_loss = torch.zeros(len(target_idx), model.num_experts, device=device)
        channel_cnt = torch.zeros(len(target_idx), device=device)
        channel_loss_events = []
        num_rule_t_values = int(max(1, config.get("phase0_rule_t_values", 5)))
        rule_t_values = torch.linspace(
            0,
            scheduler.num_timesteps - 1,
            steps=num_rule_t_values,
            device=device,
        ).round().long()
        print(
            f"   [RuleTable] Evaluating {len(eval_ds)} calibration windows "
            f"x {len(rule_t_values)} diffusion timesteps."
        )
        with torch.no_grad():
            for idx in range(len(eval_ds)):
                past, true_future, coarse_label, known_future = unpack_sample(eval_ds[idx], device)
                cl = int(coarse_label.flatten()[0].item()) if coarse_label is not None else 0
                cl = max(0, min(3, cl))
                known_future_b = known_future.unsqueeze(0) if known_future is not None else None
                condition_context = model._build_condition_context(
                    past.unsqueeze(0),
                    known_future=known_future_b,
                )
                raw_context = model._align_tensor(
                    past.unsqueeze(0), model.num_channels, resize="none"
                )
                for t_value in rule_t_values:
                    t_val = torch.full((1,), int(t_value.item()), device=device, dtype=torch.long)
                    diffusion_target = _make_diffusion_target(
                        past.unsqueeze(0),
                        true_future.unsqueeze(0),
                        forecast_objective,
                        seasonal_period,
                    ).squeeze(0)
                    noise = torch.randn_like(diffusion_target)
                    noisy = scheduler.add_noise(diffusion_target.unsqueeze(0), noise.unsqueeze(0), t_val)
                    event_channel_loss = torch.zeros(len(target_idx), model.num_experts, device=device)
                    calib_ref = noise.unsqueeze(0) if prediction_type == "epsilon" else diffusion_target.unsqueeze(0)
                    for e_idx in range(model.num_experts):
                        expert = model.experts[e_idx]
                        expert_context = (
                            raw_context
                            if getattr(expert, "uses_full_context", False)
                            else condition_context
                        )
                        pred = expert(noisy, t_val, expert_context)
                        per_loss[cl, e_idx] += F.mse_loss(pred, calib_ref).item() / len(rule_t_values)
                        for task_i, ch in enumerate(target_idx):
                            mse = F.mse_loss(pred[:, int(ch), :], calib_ref[:, int(ch), :])
                            event_channel_loss[task_i, e_idx] = mse
                            channel_loss[task_i, e_idx] += mse.item() / len(rule_t_values)
                    channel_loss_events.append(event_channel_loss.detach().cpu().numpy())
                per_cnt[cl] += 1
                channel_cnt += 1
        avg_loss = per_loss / per_cnt.unsqueeze(1).clamp_min(1)
        new_table = F.softmax(-avg_loss / 0.1, dim=-1)
        model.mmoe_gating.rule_router.rule_table.copy_(new_table)
        print(f"   [RuleTable] Updated:")
        for r in range(4):
            w = ', '.join(f'{n[:8]}:{float(new_table[r,i]):.3f}'
                          for i, n in enumerate(model.expert_names))
            print(f"     Label {r}: {w}")
        if bool(config.get("use_channel_expert_prior", True)):
            avg_channel_loss = channel_loss / channel_cnt.unsqueeze(1).clamp_min(1)
            prior_temperature = float(max(1e-6, config.get("phase0_prior_temperature", 0.1)))
            channel_prior = F.softmax(-avg_channel_loss / prior_temperature, dim=-1)
            channel_win_rate_np = np.zeros((len(target_idx), model.num_experts), dtype=np.float32)
            channel_relative_margin_np = np.zeros(len(target_idx), dtype=np.float32)
            channel_prior_confidence_np = np.zeros(len(target_idx), dtype=np.float32)
            if channel_loss_events:
                event_losses_np = np.stack(channel_loss_events, axis=0)
                winners_np = np.argmin(event_losses_np, axis=-1)
                for task_i in range(len(target_idx)):
                    for expert_i in range(model.num_experts):
                        channel_win_rate_np[task_i, expert_i] = float(np.mean(winners_np[:, task_i] == expert_i))

                avg_channel_loss_np = avg_channel_loss.detach().cpu().numpy()
                best_indices_np = np.argmin(avg_channel_loss_np, axis=1)
                winrate_threshold = float(config.get("phase0_prior_winrate_threshold", 0.6))
                winrate_threshold = max(0.0, min(1.0, winrate_threshold))
                margin_threshold = float(max(0.0, config.get("phase0_prior_margin", 0.05)))
                hard_mix = float(config.get("phase0_prior_hard_mix", 0.7))
                hard_mix = max(0.0, min(1.0, hard_mix))
                prior_np = channel_prior.detach().cpu().numpy()

                for task_i, best_idx in enumerate(best_indices_np.tolist()):
                    order = np.argsort(avg_channel_loss_np[task_i])
                    best_loss = float(avg_channel_loss_np[task_i, order[0]])
                    second_loss = float(avg_channel_loss_np[task_i, order[1]]) if len(order) > 1 else best_loss
                    denom = max(abs(second_loss), 1e-8)
                    relative_margin = max(0.0, (second_loss - best_loss) / denom)
                    win_rate = float(channel_win_rate_np[task_i, int(best_idx)])
                    channel_relative_margin_np[task_i] = relative_margin
                    if win_rate >= winrate_threshold and relative_margin >= margin_threshold:
                        win_conf = (
                            1.0
                            if winrate_threshold >= 1.0
                            else (win_rate - winrate_threshold) / max(1.0 - winrate_threshold, 1e-8)
                        )
                        margin_conf = (
                            1.0
                            if margin_threshold <= 0.0
                            else min(1.0, relative_margin / margin_threshold)
                        )
                        confidence = max(0.0, min(1.0, win_conf * margin_conf))
                        channel_prior_confidence_np[task_i] = confidence
                        one_hot = np.zeros(model.num_experts, dtype=np.float32)
                        one_hot[int(best_idx)] = 1.0
                        mix = hard_mix * confidence
                        prior_np[task_i] = (1.0 - mix) * prior_np[task_i] + mix * one_hot

                prior_np = prior_np / np.clip(prior_np.sum(axis=1, keepdims=True), 1e-8, None)
                channel_prior = torch.as_tensor(prior_np, dtype=avg_channel_loss.dtype, device=device)
            model.mmoe_gating.set_channel_expert_prior(channel_prior)
            channel_prior_np = channel_prior.detach().cpu().numpy()
            channel_loss_np = avg_channel_loss.detach().cpu().numpy()
            np.save(os.path.join(save_dir, "channel_expert_prior.npy"), channel_prior_np)
            with open(os.path.join(save_dir, "channel_expert_prior.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "target_names": list(target_names),
                        "expert_names": list(model.expert_names),
                        "prior": channel_prior_np.tolist(),
                        "avg_mse_loss": channel_loss_np.tolist(),
                        "win_rate": channel_win_rate_np.tolist(),
                        "relative_margin": channel_relative_margin_np.tolist(),
                        "prior_confidence": channel_prior_confidence_np.tolist(),
                        "num_calibration_windows": int(len(eval_ds)),
                        "num_diffusion_t_values": int(len(rule_t_values)),
                        "prior_temperature": float(prior_temperature),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            prior_rows = []
            for task_i, task_name in enumerate(target_names):
                order = np.argsort(channel_prior_np[task_i])[::-1]
                top_idx = int(order[0])
                print(
                    f"     Channel {task_name}: "
                    f"{model.expert_names[top_idx]} ({float(channel_prior_np[task_i, top_idx]):.3f})"
                )
                for expert_i, expert_name in enumerate(model.expert_names):
                    prior_rows.append(
                        {
                            # channel_idx is the real column index; channel_pos is the
                            # position inside the target list (kept for back-compat).
                            "channel_idx": int(target_idx[task_i]),
                            "channel_pos": int(task_i),
                            "channel_name": str(task_name),
                            "expert_idx": int(expert_i),
                            "expert_name": str(expert_name),
                            "prior_weight": float(channel_prior_np[task_i, expert_i]),
                            "avg_mse_loss": float(channel_loss_np[task_i, expert_i]),
                            "win_rate": float(channel_win_rate_np[task_i, expert_i]),
                            "relative_margin": float(channel_relative_margin_np[task_i]),
                            "prior_confidence": float(channel_prior_confidence_np[task_i]),
                        }
                    )
            _write_csv_rows(
                os.path.join(save_dir, "channel_expert_prior.csv"),
                prior_rows,
                [
                    "channel_idx",
                    "channel_pos",
                    "channel_name",
                    "expert_idx",
                    "expert_name",
                    "prior_weight",
                    "avg_mse_loss",
                    "win_rate",
                    "relative_margin",
                    "prior_confidence",
                ],
            )
        model.train()
    # ── End Phase 0 ─────────────────────────────────────────

    # Hard routing gives non-selected experts exactly zero gradient, so using it
    # from the start collapses the gate onto whichever expert is best after
    # Phase 0 (observed on ETTh1: every channel routed to UNet). Phase 1 therefore
    # trains with top-k soft routing; Phase 2 switches back to the hard contract.
    phase1_soft_routing = bool(config.get("phase1_soft_routing", True))
    if phase1_soft_routing:
        model.mmoe_gating.hard_final_routing = False
        model.mmoe_gating.final_top_k = int(config.get("phase1_top_k", 3))
        print(
            f"   [Routing] Phase 1 uses soft top-{int(config.get('phase1_top_k', 3))} routing "
            "so every selected expert receives gradient."
        )

    print(f"\n👉 [Phase 1] 专家混战 + 层次门控协同学习...")
    model.train()
    # Cheap validation-based early stopping. Only in deterministic inference
    # mode, where a val pass costs one forward per window rather than a full
    # diffusion trajectory. Without it the experts memorise the training split
    # (observed gap: train residual RMSE 0.21 vs val 1.0).
    phase1_early_stop = bool(config.get("phase1_early_stop", True)) and str(
        config.get("inference_mode", "diffusion")
    ) == "direct"
    phase1_val_indices = select_eval_indices(
        len(val_ds), min(int(config.get("phase1_val_windows", 128)), len(val_ds)), seed=eval_index_seed + 5
    ) if phase1_early_stop else []
    best_phase1_rmse = float("inf")
    best_phase1_state = None
    if phase1_early_stop:
        print(f"   [Phase1] early stopping on {len(phase1_val_indices)} validation windows")

    for epoch in range(config["phase1_epochs"]):
        epoch_loss = 0.0
        recon_running = 0.0
        balance_running = 0.0
        entropy_running = 0.0
        min_usage_running = 0.0
        channel_prior_running = 0.0
        aux_running = 0.0
        channel_recon_running = np.zeros(len(target_names), dtype=np.float64)
        last_batch = None
        last_label = None
        last_known_future = None

        for batch in expert_loader:
            past, future, coarse_label, known_future = unpack_batch(batch, device)
            last_batch = past
            last_label = coarse_label
            last_known_future = known_future

            diffusion_target = _make_diffusion_target(past, future, forecast_objective, seasonal_period)
            if training_mode == "regression":
                t = torch.zeros(past.shape[0], dtype=torch.long, device=device)
                noise = torch.zeros_like(diffusion_target)
                noisy_future = torch.zeros_like(diffusion_target)
            else:
                t = torch.randint(0, 1000, (past.shape[0],), device=device).long()
                noise = torch.randn_like(diffusion_target)
                noisy_future = scheduler.add_noise(diffusion_target, noise, t)

            with make_autocast_context(device, use_amp):
                pred_noise, weight_dict = forward_diffusion_model(
                    model,
                    noisy_future,
                    t,
                    past,
                    coarse_label=coarse_label,
                    known_future=known_future,
                    return_weights=True,
                )

                aux_reference = noise if prediction_type == "epsilon" else diffusion_target
                loss, loss_parts = _compute_moe_loss_components(
                    model,
                    pred_noise,
                    aux_reference,
                    target_idx,
                    weight_dict,
                    config,
                    target_loss_weights=target_loss_weights,
                )

                aux_weight = float(config.get("lambda_expert_aux", 0.0))
                expert_outputs = weight_dict.get("expert_outputs")
                if aux_weight > 0 and expert_outputs is not None:
                    # Each expert must be a good *standalone* predictor because
                    # inference commits to a single expert per channel.
                    aux_loss = F.mse_loss(
                        expert_outputs[:, :, target_idx, :],
                        aux_reference.unsqueeze(1).expand(
                            -1, expert_outputs.size(1), -1, -1
                        ),
                    )
                    loss = loss + aux_weight * aux_loss
                    loss_parts["expert_aux_loss"] = float(aux_loss.detach().item())

            optimizer_backward_step(loss, optimizer, phase1_scaler, grad_clip_norm=grad_clip_norm)

            epoch_loss += loss.item()
            recon_running += loss_parts["recon_loss"]
            balance_running += loss_parts["balance_loss"]
            entropy_running += loss_parts["entropy_loss"]
            min_usage_running += loss_parts["min_usage_loss"]
            channel_prior_running += loss_parts.get("channel_prior_loss", 0.0)
            aux_running += loss_parts.get("expert_aux_loss", 0.0)
            channel_recon_running += np.asarray(
                loss_parts.get("channel_recon_mse", np.zeros(len(target_names))),
                dtype=np.float64,
            )

        denom = max(1, len(expert_loader))
        channel_recon_avg = channel_recon_running / denom
        channel_info = " | ".join(
            f"{name}:{value:.4f}" for name, value in zip(target_names, channel_recon_avg)
        )
        print(
            f"   [P1] Epoch {epoch+1} | Loss: {epoch_loss/denom:.6f} | "
            f"Recon: {recon_running/denom:.6f} | "
            f"Balance: {balance_running/denom:.6f} | "
            f"Entropy: {entropy_running/denom:.6f} | "
            f"MinUsage: {min_usage_running/denom:.6f} | "
            f"ChannelPrior: {channel_prior_running/denom:.6f} | "
            f"Aux: {aux_running/denom:.6f}"
        )
        print(f"   [P1] Channel Recon MSE: {channel_info}")
        for channel_pos, (channel_name, channel_value) in enumerate(zip(target_names, channel_recon_avg)):
            channel_train_loss_rows.append(
                {
                    "phase": "phase1",
                    "epoch": int(epoch + 1),
                    "channel_pos": int(channel_pos),
                    "channel_idx": int(target_idx[channel_pos]),
                    "channel_name": str(channel_name),
                    "recon_mse": float(channel_value),
                }
            )

        if (epoch + 1) % 5 == 0 and last_batch is not None:
            log_moe_status(
                model,
                last_batch,
                epoch + 1,
                "P1",
                coarse_label_batch=last_label,
                known_future_batch=last_known_future,
                task_names=target_names,
            )

        if phase1_early_stop and ((epoch + 1) % max(1, int(config.get("phase1_val_every", 2))) == 0):
            model.eval()
            with torch.no_grad():
                val_metrics, _, _ = _evaluate_subset_metrics(
                    model,
                    scheduler,
                    val_ds,
                    device,
                    phase1_val_indices,
                    target_idx=target_idx,
                    pred_len=int(config.get("pred_len", 96)),
                    horizon_ranges=resolve_horizon_ranges(
                        int(config.get("pred_len", 96)), horizon_boundaries
                    ),
                    sample_sampler=eval_sampler,
                    sample_inference_steps=eval_inference_steps,
                    sample_eta=eval_eta,
                    sample_t_start=config.get("ddim_t_start", None),
                    strict_sample_shape=strict_sample_shape,
                    use_amp=use_amp,
                    drift_correction=bool(config.get("drift_correction", True)),
                    drift_tail_window=int(config.get("drift_tail_window", 24)),
                    drift_alpha=float(config.get("drift_alpha", 0.75)),
                    sample_seed=eval_sample_seed,
                    batch_size=eval_batch_size,
                    forecast_objective=forecast_objective,
                    seasonal_period=seasonal_period,
                    eval_space=str(config.get("eval_space", "raw")),
                    inference_mode=str(config.get("inference_mode", "diffusion")),
                )
            model.train()
            if val_metrics is not None:
                phase1_rmse = float(val_metrics["RMSE"])
                if phase1_rmse < best_phase1_rmse:
                    best_phase1_rmse = phase1_rmse
                    best_phase1_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    print(f"   [Phase1] val RMSE={phase1_rmse:.4f} (best so far)")

    if best_phase1_state is not None:
        model.load_state_dict(best_phase1_state)
        print(f"   [Phase1] restored best checkpoint | val RMSE={best_phase1_rmse:.4f}")

    if phase1_soft_routing:
        model.mmoe_gating.hard_final_routing = bool(config.get("hard_final_routing", True))
        model.mmoe_gating.final_top_k = int(config.get("final_top_k", 1))
        print(
            f"   [Routing] Phase 2 restored hard routing "
            f"(hard={bool(config.get('hard_final_routing', True))}, top_k={int(config.get('final_top_k', 1))})."
        )

    print(f"\n👉 [Phase 2] 冻结专家，训练粗/细粒度门控...")
    model.freeze_experts()
    gate_params = list(model.mmoe_gating.parameters())
    if getattr(model, "use_task_heads", False):
        gate_params.extend(list(model.task_heads.parameters()))
    gate_optim = torch.optim.Adam(gate_params, lr=1e-4, weight_decay=1e-5)

    # Keep the best Phase-1 weights as the fallback: Phase 2 only trains the
    # gate, and if it cannot beat the joint model we must not ship a worse one.
    best_rmse = float(best_phase1_rmse) if best_phase1_state is not None else float("inf")
    best_epoch = None
    patience = 0
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    for epoch in range(config["phase2_epochs"]):
        model.train()
        gate_loss = 0.0
        recon_running = 0.0
        balance_running = 0.0
        entropy_running = 0.0
        min_usage_running = 0.0
        channel_prior_running = 0.0
        channel_recon_running = np.zeros(len(target_names), dtype=np.float64)
        last_batch = None
        last_label = None
        last_known_future = None

        for batch in gate_loader:
            past, future, coarse_label, known_future = unpack_batch(batch, device)
            last_batch = past
            last_label = coarse_label
            last_known_future = known_future

            diffusion_target = _make_diffusion_target(past, future, forecast_objective, seasonal_period)
            if training_mode == "regression":
                t = torch.zeros(past.shape[0], dtype=torch.long, device=device)
                noise = torch.zeros_like(diffusion_target)
                noisy_future = torch.zeros_like(diffusion_target)
            else:
                t = torch.randint(0, 1000, (past.shape[0],), device=device).long()
                noise = torch.randn_like(diffusion_target)
                noisy_future = scheduler.add_noise(diffusion_target, noise, t)

            with make_autocast_context(device, use_amp):
                pred_noise, weight_dict = forward_diffusion_model(
                    model,
                    noisy_future,
                    t,
                    past,
                    coarse_label=coarse_label,
                    known_future=known_future,
                    return_weights=True,
                )

                loss, loss_parts = _compute_moe_loss_components(
                    model,
                    pred_noise,
                    noise if prediction_type == "epsilon" else diffusion_target,
                    target_idx,
                    weight_dict,
                    config,
                    target_loss_weights=target_loss_weights,
                )

            optimizer_backward_step(loss, gate_optim, phase2_scaler, grad_clip_norm=grad_clip_norm)

            gate_loss += loss.item()
            recon_running += loss_parts["recon_loss"]
            balance_running += loss_parts["balance_loss"]
            entropy_running += loss_parts["entropy_loss"]
            min_usage_running += loss_parts["min_usage_loss"]
            channel_prior_running += loss_parts.get("channel_prior_loss", 0.0)
            channel_recon_running += np.asarray(
                loss_parts.get("channel_recon_mse", np.zeros(len(target_names))),
                dtype=np.float64,
            )

        denom = max(1, len(gate_loader))
        avg_loss = gate_loss / denom
        channel_recon_avg = channel_recon_running / denom
        channel_info = " | ".join(
            f"{name}:{value:.4f}" for name, value in zip(target_names, channel_recon_avg)
        )
        print(
            f"   [P2] Epoch {epoch+1} | Gate Loss: {avg_loss:.6f} | "
            f"Recon: {recon_running/denom:.6f} | "
            f"Balance: {balance_running/denom:.6f} | "
            f"Entropy: {entropy_running/denom:.6f} | "
            f"MinUsage: {min_usage_running/denom:.6f} | "
            f"ChannelPrior: {channel_prior_running/denom:.6f}"
        )
        print(f"   [P2] Channel Recon MSE: {channel_info}")
        for channel_pos, (channel_name, channel_value) in enumerate(zip(target_names, channel_recon_avg)):
            channel_train_loss_rows.append(
                {
                    "phase": "phase2",
                    "epoch": int(epoch + 1),
                    "channel_pos": int(channel_pos),
                    "channel_idx": int(target_idx[channel_pos]),
                    "channel_name": str(channel_name),
                    "recon_mse": float(channel_value),
                }
            )

        if last_batch is not None:
            log_moe_status(
                model,
                last_batch,
                epoch + 1,
                "P2",
                coarse_label_batch=last_label,
                known_future_batch=last_known_future,
                task_names=target_names,
            )

        should_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == config["phase2_epochs"])
        if should_eval:
            metrics = run_evaluation(
                model,
                scheduler,
                val_ds,
                f"P2_Ep{epoch+1}",
                device,
                save_dir,
                export_all_windows=False,
                max_eval_windows=val_metric_num_samples,
                index_seed=eval_index_seed,
                sample_seed=eval_sample_seed,
                save_preview_plot=True,
                save_window_plots=False,
                save_overview_plots=False,
                sample_sampler=eval_sampler,
                sample_inference_steps=eval_inference_steps,
                sample_eta=eval_eta,
                sample_t_start=config.get("ddim_t_start", None),
                use_amp=use_amp,
                drift_correction=bool(config.get("drift_correction", True)),
                drift_tail_window=int(config.get("drift_tail_window", 24)),
                drift_alpha=float(config.get("drift_alpha", 0.75)),
                horizon_boundaries=horizon_boundaries,
                save_expert_health=True,
                expert_health_max_windows=int(config.get("expert_health_max_windows", 16)),
                crps_samples=int(config.get("crps_samples", 0)),
                crps_num_windows=int(config.get("crps_num_windows", 16)),
                eval_batch_size=eval_batch_size,
                strict_sample_shape=strict_sample_shape,
                forecast_objective=forecast_objective,
                seasonal_period=seasonal_period,
                baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
                eval_space=str(config.get("eval_space", "raw")),
                eval_ensemble_samples=int(config.get("eval_ensemble_samples", 1)),
                inference_mode=str(config.get("inference_mode", "diffusion")),
            )

            current_rmse = float(metrics["RMSE"]) if metrics is not None else float("inf")
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_epoch = int(epoch + 1)
                patience = 0
                torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
                with open(os.path.join(save_dir, "best_val_metric.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "selection_metric": "RMSE",
                            "selection_split": "val_subset",
                            "subset_windows": int(val_metric_num_samples),
                            "sampler": eval_sampler,
                            "inference_steps": int(eval_inference_steps),
                            "phase": "phase2",
                            "epoch": best_epoch,
                            "RMSE": best_rmse,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"   [Best] Phase2 Epoch {best_epoch} | ValSubset RMSE={best_rmse:.4f}")
            else:
                patience += 1
                if patience >= 5:
                    print("Early stopping: phase2 validation metric did not improve.")
                    break

            if metrics is not None and metrics.get("expert_health_rows") is not None:
                maybe_update_expert_lifecycle(
                    model,
                    metrics.get("expert_health_rows"),
                    epoch=epoch + 1,
                    phase="phase2",
                    config=config,
                    save_dir=save_dir,
                    channel_expert_rows=metrics.get("channel_expert_health_rows"),
                )

    if best_epoch is None and not os.path.exists(os.path.join(save_dir, "model_weights.pth")):
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    if channel_train_loss_rows:
        _write_csv_rows(
            os.path.join(save_dir, "channel_train_loss.csv"),
            channel_train_loss_rows,
            ["phase", "epoch", "channel_pos", "channel_idx", "channel_name", "recon_mse"],
        )

    model.load_state_dict(torch.load(os.path.join(save_dir, "model_weights.pth"), map_location=device))
    final_metrics = run_evaluation(
        model,
        scheduler,
        val_ds,
        "Final",
        device,
        save_dir,
        export_all_windows=final_eval_export_all_windows,
        max_eval_windows=final_eval_num_samples,
        index_seed=eval_index_seed,
        sample_seed=eval_sample_seed,
        save_window_plots=bool(config.get("final_eval_save_window_plots", False)),
        window_plot_limit=config.get("final_eval_window_plot_limit", 0),
        save_overview_plots=bool(config.get("final_eval_save_overview_plots", True)),
        sample_sampler=final_eval_sampler,
        sample_inference_steps=final_eval_inference_steps,
        sample_eta=final_eval_eta,
        sample_t_start=config.get("ddim_t_start", None),
        use_amp=use_amp,
        drift_correction=bool(config.get("drift_correction", True)),
        drift_tail_window=int(config.get("drift_tail_window", 24)),
        drift_alpha=float(config.get("drift_alpha", 0.75)),
        horizon_boundaries=horizon_boundaries,
        save_expert_health=True,
        expert_health_max_windows=int(
            config.get(
                "expert_health_max_windows",
                final_eval_num_samples if final_eval_num_samples is not None else val_metric_num_samples,
            )
        ),
        crps_samples=int(config.get("crps_samples", 0)),
        crps_num_windows=int(config.get("crps_num_windows", 16)),
        eval_batch_size=eval_batch_size,
        strict_sample_shape=strict_sample_shape,
        forecast_objective=forecast_objective,
        seasonal_period=seasonal_period,
        baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
        eval_space=str(config.get("eval_space", "raw")),
        eval_ensemble_samples=int(config.get("eval_ensemble_samples", 1)),
        inference_mode=str(config.get("inference_mode", "diffusion")),
    )

    # Held-out test split: the validation metrics above are used for early
    # stopping, so the paper's headline numbers must come from a split the
    # model never selected on.
    if test_ds is not None and bool(config.get("eval_on_test", True)) and len(test_ds) > 0:
        test_export_all = bool(config.get("final_test_export_all_windows", True))
        test_num_samples = None
        if not test_export_all:
            test_num_samples = int(
                config.get("final_test_num_samples", val_metric_num_samples)
            )
        print(f"\n👉 [Test] Held-out test evaluation ({len(test_ds)} windows)...")
        final_test_metrics = run_evaluation(
            model,
            scheduler,
            test_ds,
            "FinalTest",
            device,
            save_dir,
            export_all_windows=test_export_all,
            max_eval_windows=test_num_samples,
            index_seed=eval_index_seed + 3,
            sample_seed=eval_sample_seed,
            save_preview_plot=False,
            save_window_plots=False,
            save_overview_plots=False,
            sample_sampler=final_eval_sampler,
            sample_inference_steps=final_eval_inference_steps,
            sample_eta=final_eval_eta,
            sample_t_start=config.get("ddim_t_start", None),
            use_amp=use_amp,
            drift_correction=bool(config.get("drift_correction", True)),
            drift_tail_window=int(config.get("drift_tail_window", 24)),
            drift_alpha=float(config.get("drift_alpha", 0.75)),
            horizon_boundaries=horizon_boundaries,
            save_expert_health=False,
            crps_samples=int(config.get("crps_samples", 0)),
            crps_num_windows=int(config.get("crps_num_windows", 16)),
            eval_batch_size=eval_batch_size,
            strict_sample_shape=strict_sample_shape,
            forecast_objective=forecast_objective,
            seasonal_period=seasonal_period,
            baseline_reference=str(config.get("baseline_reference", "daily_seasonal")),
            eval_space=str(config.get("eval_space", "raw")),
            eval_ensemble_samples=int(config.get("eval_ensemble_samples", 1)),
            inference_mode=str(config.get("inference_mode", "diffusion")),
        )
        if final_test_metrics is not None:
            print(
                f"   [Test] RMSE={float(final_test_metrics['RMSE']):.4f} | "
                f"MAE={float(final_test_metrics['MAE']):.4f} | R2={float(final_test_metrics['R2']):.4f}"
            )
        selection_path = os.path.join(save_dir, "best_val_metric.json")
        selection_payload = {}
        if os.path.exists(selection_path):
            with open(selection_path, "r", encoding="utf-8") as f:
                selection_payload = json.load(f)
        selection_payload["test_rmse"] = None if final_test_metrics is None else float(final_test_metrics["RMSE"])
        selection_payload["test_r2"] = None if final_test_metrics is None else float(final_test_metrics["R2"])
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, indent=2, ensure_ascii=False)

    best_payload = _export_channel_best_expert(save_dir, target_names, model.expert_names, final_metrics)
    if best_payload is not None:
        print("✅ Saved channel_best_expert.json / channel_best_expert.csv")
    export_moe_logs(model, val_ds, device, save_dir)


def export_moe_logs(model, dataset, device, save_dir):
    print("📝 导出层次化 MoE 门控权重日志...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_weights.pth"), map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    final_logs, coarse_logs, fine_logs = [], [], []

    with torch.no_grad():
        for batch in loader:
            past, _, coarse_label, known_future = unpack_batch(batch, device)
            _, weight_dict = forward_diffusion_model(
                model,
                None,
                None,
                past,
                coarse_label=coarse_label,
                known_future=known_future,
                return_weights=True,
            )
            final_logs.append(weight_dict["final"].cpu().numpy())
            coarse_logs.append(weight_dict["coarse"].cpu().numpy())
            fine_logs.append(weight_dict["fine"].cpu().numpy())

    final_arr = np.concatenate(final_logs, axis=0)
    coarse_arr = np.concatenate(coarse_logs, axis=0)
    fine_arr = np.concatenate(fine_logs, axis=0)
    if coarse_arr.ndim == 2:
        coarse_arr = np.repeat(coarse_arr[:, None, :], final_arr.shape[1], axis=1)

    np.save(os.path.join(save_dir, "gate_final.npy"), final_arr)
    np.save(os.path.join(save_dir, "gate_coarse.npy"), coarse_arr)
    np.save(os.path.join(save_dir, "gate_fine.npy"), fine_arr)

    task_names = list(getattr(dataset, "target_cols", []) or [])
    if len(task_names) != final_arr.shape[1]:
        task_names = [f"task{i}" for i in range(final_arr.shape[1])]
    with open(os.path.join(save_dir, "gate_task_names.json"), "w", encoding="utf-8") as f:
        json.dump(task_names, f, indent=2, ensure_ascii=False)

    rows = []
    selected = final_arr > 0
    top1 = np.argmax(final_arr, axis=-1)
    for task_i, task_name in enumerate(task_names):
        for expert_i, expert_name in enumerate(model.expert_names):
            rows.append(
                {
                    "task_idx": int(task_i),
                    "task_name": str(task_name),
                    "expert_idx": int(expert_i),
                    "expert_name": str(expert_name),
                    "avg_final_weight": float(final_arr[:, task_i, expert_i].mean()),
                    "selection_freq": float(selected[:, task_i, expert_i].mean()),
                    "avg_coarse_weight": float(coarse_arr[:, task_i, expert_i].mean()),
                    "avg_fine_weight": float(fine_arr[:, task_i, expert_i].mean()),
                    "selected_freq": float(selected[:, task_i, expert_i].mean()),
                    "top1_freq": float((top1[:, task_i] == expert_i).mean()),
                }
            )
    _write_csv_rows(
        os.path.join(save_dir, "gate_task_summary.csv"),
        rows,
        [
            "task_idx",
            "task_name",
            "expert_idx",
            "expert_name",
            "avg_final_weight",
            "selection_freq",
            "avg_coarse_weight",
            "avg_fine_weight",
            "selected_freq",
            "top1_freq",
        ],
    )

    print("✅ 已保存: gate_final.npy / gate_coarse.npy / gate_fine.npy / gate_task_summary.csv")
