import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.metrics import MetricsCalculator, resolve_horizon_ranges
from utils.postprocess import apply_linear_drift_correction
from network.moe import MoEDiffuser
from training.core import forward_diffusion_model, unpack_batch, unpack_sample
from training.common import (
    _ProgressReporter,
    _append_jsonl,
    _bcl_to_blc,
    _compute_target_recon_loss,
    _distribution_entropy,
    _jsonable_metrics,
    _temporary_expert_gate_scale,
    make_autocast_context,
)
from training.baselines import (
    normalize_forecast_objective,
    restore_residual_forecast_bcl,
    uses_residual_forecast_objective,
)


def _evaluate_subset_metrics(
    model,
    scheduler,
    dataset,
    device,
    indices,
    target_idx,
    pred_len: int,
    horizon_ranges,
    sample_sampler: str = "ddpm",
    sample_inference_steps: int | None = None,
    sample_eta: float = 0.0,
    sample_t_start: int | None = None,
    strict_sample_shape: bool = True,
    use_amp: bool = False,
    drift_correction: bool = False,
    drift_tail_window: int = 24,
    drift_alpha: float = 0.75,
    sample_seed: int | None = None,
    batch_size: int = 1,
    forecast_objective: str = "direct",
    seasonal_period: int = 24,
    eval_space: str = "raw",
    inference_mode: str = "diffusion",
):
    clean_indices = [int(i) for i in (indices or [])]
    if len(clean_indices) == 0:
        return None, [], []
    batch_size = int(max(1, batch_size))
    forecast_objective = normalize_forecast_objective(forecast_objective)
    seasonal_period = int(max(1, seasonal_period))
    residual_objective = uses_residual_forecast_objective(forecast_objective)
    use_scaled_space = str(eval_space or "raw").strip().lower() == "scaled"

    trues_list, preds_list = [], []
    horizon_true = {label: [] for label, _, _ in horizon_ranges}
    horizon_pred = {label: [] for label, _, _ in horizon_ranges}
    channel_true = [[] for _ in target_idx]
    channel_pred = [[] for _ in target_idx]

    for batch_start in range(0, len(clean_indices), batch_size):
        batch_indices = clean_indices[batch_start: batch_start + batch_size]
        batch_items = [unpack_sample(dataset[idx], device) for idx in batch_indices]

        past_bcl = torch.stack([item[0] for item in batch_items], dim=0)
        true_bcl = torch.stack([item[1] for item in batch_items], dim=0)

        coarse_items = [item[2] for item in batch_items]
        if any(label is not None for label in coarse_items):
            template_label = next(v for v in coarse_items if v is not None)
            coarse_label_b = torch.stack(
                [label if label is not None else torch.zeros_like(template_label) for label in coarse_items],
                dim=0,
            )
        else:
            coarse_label_b = None

        known_items = [item[3] for item in batch_items]
        if any(known is not None and known.numel() > 0 for known in known_items):
            template_known = next(v for v in known_items if v is not None and v.numel() > 0)
            # Missing known covariates are padded as zero by contract; KnownFutureAdapter is zero-initialized.
            known_future_b = torch.stack(
                [known if known is not None else torch.zeros_like(template_known) for known in known_items],
                dim=0,
            )
        else:
            known_future_b = None

        with torch.no_grad(), make_autocast_context(device, use_amp):
            sample_seed_i = None if sample_seed is None else int(sample_seed) + int(batch_indices[0])
            if inference_mode == "direct":
                gen_bcl = model(
                    torch.zeros_like(past_bcl),
                    torch.zeros(past_bcl.size(0), dtype=torch.long, device=past_bcl.device),
                    past_bcl,
                    coarse_label=coarse_label_b,
                    known_future=known_future_b,
                    return_weights=False,
                )
                if isinstance(gen_bcl, tuple):
                    gen_bcl = gen_bcl[0]
            else:
                gen_bcl = scheduler.sample(
                model,
                past_bcl,
                pred_len,
                coarse_label=coarse_label_b,
                known_future=known_future_b,
                seed=sample_seed_i,
                sampler=sample_sampler,
                num_inference_steps=sample_inference_steps,
                eta=sample_eta,
                t_start=sample_t_start,
                strict_seq_len=strict_sample_shape,
            )
            if residual_objective:
                gen_bcl = restore_residual_forecast_bcl(
                    past_bcl,
                    gen_bcl,
                    period=seasonal_period,
                )

        past_blc = _bcl_to_blc(past_bcl)
        true_blc = _bcl_to_blc(true_bcl)
        pred_blc = _bcl_to_blc(gen_bcl)

        for local_i, _idx in enumerate(batch_indices):
            past_lc = past_blc[local_i]
            true_lc = true_blc[local_i]
            pred_lc = pred_blc[local_i]

            if use_scaled_space:
                past_raw, true_raw, pred_raw = past_lc, true_lc, pred_lc
            elif hasattr(dataset, "inverse_transform"):
                past_raw = dataset.inverse_transform(past_lc)
                true_raw = dataset.inverse_transform(true_lc)
                pred_raw = dataset.inverse_transform(pred_lc)
            else:
                past_raw, true_raw, pred_raw = past_lc, true_lc, pred_lc

            if drift_correction:
                pred_raw = apply_linear_drift_correction(
                    past_raw,
                    pred_raw,
                    target_indices=target_idx,
                    tail_window=drift_tail_window,
                    alpha=drift_alpha,
                )

            true_t = np.asarray(true_raw)[:, target_idx]
            pred_t = np.asarray(pred_raw)[:, target_idx]
            trues_list.extend(true_t.reshape(-1).tolist())
            preds_list.extend(pred_t.reshape(-1).tolist())
            for channel_pos in range(len(target_idx)):
                channel_true[channel_pos].extend(true_t[:, channel_pos].reshape(-1).tolist())
                channel_pred[channel_pos].extend(pred_t[:, channel_pos].reshape(-1).tolist())

            for label, start_h, end_h in horizon_ranges:
                horizon_true[label].extend(true_t[start_h:end_h, :].reshape(-1).tolist())
                horizon_pred[label].extend(pred_t[start_h:end_h, :].reshape(-1).tolist())

    metrics = _jsonable_metrics(MetricsCalculator.compute(
        np.nan_to_num(np.asarray(trues_list, dtype=np.float64), nan=0.0),
        np.nan_to_num(np.asarray(preds_list, dtype=np.float64), nan=0.0),
    ))
    horizon_rows = []
    for label, start_h, end_h in horizon_ranges:
        if len(horizon_true[label]) == 0:
            continue
        h_metrics = _jsonable_metrics(MetricsCalculator.compute(
            np.nan_to_num(np.asarray(horizon_true[label], dtype=np.float64), nan=0.0),
            np.nan_to_num(np.asarray(horizon_pred[label], dtype=np.float64), nan=0.0),
        ))
        horizon_rows.append({
            "horizon": str(label),
            "start_step": int(start_h + 1),
            "end_step": int(end_h),
            "MAE": h_metrics["MAE"],
            "MSE": h_metrics["MSE"],
            "RMSE": h_metrics["RMSE"],
            "MAPE": h_metrics["MAPE"],
            "MSPE": h_metrics["MSPE"],
            "sMAPE": h_metrics["sMAPE"],
            "R2": h_metrics["R2"],
        })
    channel_rows = []
    for channel_pos, channel_idx in enumerate(target_idx):
        if len(channel_true[channel_pos]) == 0:
            continue
        ch_metrics = _jsonable_metrics(MetricsCalculator.compute(
            np.nan_to_num(np.asarray(channel_true[channel_pos], dtype=np.float64), nan=0.0),
            np.nan_to_num(np.asarray(channel_pred[channel_pos], dtype=np.float64), nan=0.0),
        ))
        channel_rows.append({
            "channel_idx": int(channel_idx),
            "channel_pos": int(channel_pos),
            "MAE": ch_metrics["MAE"],
            "MSE": ch_metrics["MSE"],
            "RMSE": ch_metrics["RMSE"],
            "MAPE": ch_metrics["MAPE"],
            "MSPE": ch_metrics["MSPE"],
            "sMAPE": ch_metrics["sMAPE"],
            "R2": ch_metrics["R2"],
        })
    return metrics, horizon_rows, channel_rows


def _resolve_expert_health_signal(row):
    contribution_ready = bool(row.get("contribution_evaluated", False))
    if contribution_ready:
        overall = float(row.get("rmse_delta_ratio_if_removed", 0.0))
        best_horizon = float(row.get("best_horizon_rmse_delta_ratio_if_removed", overall))
        return max(overall, best_horizon), "marginal_contribution"

    usage_signal = max(
        float(row.get("selected_freq", 0.0)),
        float(row.get("top1_freq", 0.0)),
        float(row.get("avg_final_weight", 0.0)),
    )
    return usage_signal, "usage"


def _compute_moe_loss_components(
    model,
    pred_noise,
    noise,
    target_idx,
    weight_dict,
    config,
    target_loss_weights: torch.Tensor | None = None,
):
    pred_target = pred_noise[:, target_idx, :]
    noise_target = noise[:, target_idx, :]
    channel_recon_mse = (pred_target - noise_target).pow(2).mean(dim=(0, 2))
    recon_loss = _compute_target_recon_loss(
        pred_noise,
        noise,
        target_idx,
        target_loss_weights=target_loss_weights,
    )
    balance_loss = model.gate_balance_loss(weight_dict)
    entropy_loss = model.gate_entropy_loss(weight_dict)
    min_usage_loss = model.gate_min_usage_loss(
        weight_dict,
        min_usage=float(config.get("min_expert_usage", 0.0)),
    )
    channel_prior_loss = model.channel_prior_loss(weight_dict)

    loss = recon_loss
    loss = loss + float(config.get("lambda_balance", 1e-3)) * balance_loss
    loss = loss + float(config.get("lambda_entropy", 0.0)) * entropy_loss
    loss = loss + float(config.get("lambda_min_usage", 0.0)) * min_usage_loss
    loss = loss + float(config.get("lambda_channel_prior", 0.0)) * channel_prior_loss

    return loss, {
        "recon_loss": float(recon_loss.detach().item()),
        "balance_loss": float(balance_loss.detach().item()),
        "entropy_loss": float(entropy_loss.detach().item()),
        "min_usage_loss": float(min_usage_loss.detach().item()),
        "channel_prior_loss": float(channel_prior_loss.detach().item()),
        "channel_recon_mse": [float(v) for v in channel_recon_mse.detach().cpu().tolist()],
    }


def collect_moe_expert_health(
    model,
    dataset,
    device,
    indices=None,
    contribution_indices=None,
    batch_size: int = 32,
    use_amp: bool = False,
    scheduler=None,
    sample_sampler: str = "ddpm",
    sample_inference_steps: int | None = None,
    sample_eta: float = 0.0,
    sample_t_start: int | None = None,
    strict_sample_shape: bool = True,
    sample_seed: int | None = None,
    drift_correction: bool = False,
    drift_tail_window: int = 24,
    drift_alpha: float = 0.75,
    target_idx=None,
    pred_len: int | None = None,
    horizon_ranges=None,
    forecast_objective: str = "direct",
    seasonal_period: int = 24,
    eval_space: str = "raw",
    inference_mode: str = "diffusion",
):
    if not isinstance(model, MoEDiffuser):
        return [], None, [], []

    if indices is None:
        subset = dataset
        total_items = len(dataset)
        usage_indices = list(range(len(dataset)))
    else:
        usage_indices = [int(i) for i in indices]
        subset = Subset(dataset, usage_indices)
        total_items = len(usage_indices)

    if total_items <= 0:
        return [], None, [], []

    contribution_eval_indices = usage_indices if contribution_indices is None else [int(i) for i in contribution_indices]
    if target_idx is None:
        target_idx = getattr(dataset, "target_indices", None)
    if not target_idx:
        target_idx = list(range(getattr(dataset, "num_channels", 1)))
    target_idx = [int(i) for i in target_idx]
    target_names = list(getattr(dataset, "target_cols", []) or [])
    if len(target_names) != len(target_idx):
        target_names = [f"ch{idx}" for idx in target_idx]
    if pred_len is None:
        pred_len = int(getattr(dataset, "pred_len", getattr(dataset, "seq_len", 0)))
    pred_len = int(pred_len)
    if horizon_ranges is None:
        horizon_ranges = resolve_horizon_ranges(pred_len)
    forecast_objective = normalize_forecast_objective(forecast_objective)
    seasonal_period = int(max(1, seasonal_period))

    loader = DataLoader(subset, batch_size=int(batch_size), shuffle=False)
    num_experts = int(model.num_experts)
    final_sum = np.zeros(num_experts, dtype=np.float64)
    final_dense_sum = np.zeros(num_experts, dtype=np.float64)
    coarse_sum = np.zeros(num_experts, dtype=np.float64)
    fine_sum = np.zeros(num_experts, dtype=np.float64)
    selected_count = np.zeros(num_experts, dtype=np.float64)
    coarse_selected_count = np.zeros(num_experts, dtype=np.float64)
    top1_count = np.zeros(num_experts, dtype=np.float64)
    channel_final_sum = None
    channel_final_dense_sum = None
    channel_coarse_sum = None
    channel_fine_sum = None
    channel_selected_count = None
    channel_top1_count = None
    total_windows = 0
    total_route_slots = 0

    model.eval()
    routing_progress = _ProgressReporter("Health Routing", total_items, min_interval_sec=15.0)
    routed_items = 0
    with torch.no_grad():
        for batch in loader:
            past, _, coarse_label, known_future = unpack_batch(batch, device)
            coarse_label_batch = coarse_label if coarse_label is not None else None
            with make_autocast_context(device, use_amp):
                _, weight_dict = forward_diffusion_model(
                    model,
                    None,
                    None,
                    past,
                    coarse_label=coarse_label_batch,
                    known_future=known_future,
                    return_weights=True,
                )

            final = weight_dict["final"].detach()
            final_dense = weight_dict.get("final_dense", final).detach()
            coarse = weight_dict["coarse"].detach()
            fine = weight_dict["fine"].detach()

            B, T, E = final.shape
            if coarse.dim() == 2:
                coarse = coarse.unsqueeze(1).expand(-1, T, -1)
            if channel_final_sum is None:
                channel_final_sum = np.zeros((T, E), dtype=np.float64)
                channel_final_dense_sum = np.zeros((T, E), dtype=np.float64)
                channel_coarse_sum = np.zeros((T, E), dtype=np.float64)
                channel_fine_sum = np.zeros((T, E), dtype=np.float64)
                channel_selected_count = np.zeros((T, E), dtype=np.float64)
                channel_top1_count = np.zeros((T, E), dtype=np.float64)
            total_windows += int(B)
            total_route_slots += int(B * T)
            routed_items += int(B)

            final_sum += final.sum(dim=(0, 1)).cpu().numpy()
            final_dense_sum += final_dense.sum(dim=(0, 1)).cpu().numpy()
            coarse_sum += coarse.sum(dim=(0, 1)).cpu().numpy()
            fine_sum += fine.sum(dim=(0, 1)).cpu().numpy()
            selected_count += (final > 0).sum(dim=(0, 1)).cpu().numpy()
            coarse_selected_count += (coarse > 0).sum(dim=(0, 1)).cpu().numpy()
            channel_final_sum += final.sum(dim=0).cpu().numpy()
            channel_final_dense_sum += final_dense.sum(dim=0).cpu().numpy()
            channel_coarse_sum += coarse.sum(dim=0).cpu().numpy()
            channel_fine_sum += fine.sum(dim=0).cpu().numpy()
            channel_selected_count += (final > 0).sum(dim=0).cpu().numpy()

            top1 = torch.argmax(final, dim=-1)
            top1_hist = F.one_hot(top1, num_classes=E).sum(dim=(0, 1)).cpu().numpy()
            top1_count += top1_hist
            channel_top1_count += F.one_hot(top1, num_classes=E).sum(dim=0).cpu().numpy()
            routing_progress.update(routed_items, extra=f"batch={int(B)}")

    if total_windows <= 0 or total_route_slots <= 0:
        return [], None, [], []

    avg_final = final_sum / total_route_slots
    avg_final_dense = final_dense_sum / total_route_slots
    avg_coarse = coarse_sum / total_route_slots
    avg_fine = fine_sum / total_route_slots
    selected_freq = selected_count / total_route_slots
    coarse_selected_freq = coarse_selected_count / total_route_slots
    top1_freq = top1_count / total_route_slots

    status_strings = model.get_expert_status_strings()
    gate_scale = model.expert_gate_scale.detach().cpu().numpy()
    bad_epochs = model.expert_bad_epochs.detach().cpu().numpy()
    soft_retired_epoch = model.expert_soft_retired_epoch.detach().cpu().numpy()
    hard_pruned_epoch = model.expert_hard_pruned_epoch.detach().cpu().numpy()

    rows = []
    for expert_idx, expert_name in enumerate(model.expert_names):
        rows.append(
            {
                "expert_idx": int(expert_idx),
                "expert_name": str(expert_name),
                "status": status_strings[expert_idx],
                "gate_scale": float(gate_scale[expert_idx]),
                "bad_epochs": int(bad_epochs[expert_idx]),
                "soft_retired_epoch": int(soft_retired_epoch[expert_idx]),
                "hard_pruned_epoch": int(hard_pruned_epoch[expert_idx]),
                "avg_final_weight": float(avg_final[expert_idx]),
                "avg_final_dense_weight": float(avg_final_dense[expert_idx]),
                "avg_coarse_weight": float(avg_coarse[expert_idx]),
                "avg_fine_weight": float(avg_fine[expert_idx]),
                "selected_freq": float(selected_freq[expert_idx]),
                "selection_freq": float(selected_freq[expert_idx]),
                "coarse_selected_freq": float(coarse_selected_freq[expert_idx]),
                "top1_freq": float(top1_freq[expert_idx]),
                "contribution_evaluated": False,
                "rmse_delta_if_removed": 0.0,
                "rmse_delta_ratio_if_removed": 0.0,
                "r2_delta_if_removed": 0.0,
                "best_horizon_if_removed": "",
                "best_horizon_rmse_delta_if_removed": 0.0,
                "best_horizon_rmse_delta_ratio_if_removed": 0.0,
                "worst_horizon_if_removed": "",
                "worst_horizon_rmse_delta_if_removed": 0.0,
                "worst_horizon_rmse_delta_ratio_if_removed": 0.0,
                "contribution_score": 0.0,
                "horizon_rmse_delta_if_removed_json": "{}",
            }
        )

    channel_expert_rows = []
    channel_expert_by_key = {}
    denom_channel = float(max(total_windows, 1))
    for channel_pos in range(channel_final_sum.shape[0]):
        channel_name = str(target_names[channel_pos]) if channel_pos < len(target_names) else f"ch{channel_pos}"
        channel_idx = int(target_idx[channel_pos]) if channel_pos < len(target_idx) else int(channel_pos)
        for expert_idx, expert_name in enumerate(model.expert_names):
            row = {
                "channel_idx": channel_idx,
                "channel_pos": int(channel_pos),
                "channel_name": channel_name,
                "expert_idx": int(expert_idx),
                "expert_name": str(expert_name),
                "status": status_strings[expert_idx],
                "gate_scale": float(gate_scale[expert_idx]),
                "avg_final_weight": float(channel_final_sum[channel_pos, expert_idx] / denom_channel),
                "avg_final_dense_weight": float(channel_final_dense_sum[channel_pos, expert_idx] / denom_channel),
                "avg_coarse_weight": float(channel_coarse_sum[channel_pos, expert_idx] / denom_channel),
                "avg_fine_weight": float(channel_fine_sum[channel_pos, expert_idx] / denom_channel),
                "selected_freq": float(channel_selected_count[channel_pos, expert_idx] / denom_channel),
                "selection_freq": float(channel_selected_count[channel_pos, expert_idx] / denom_channel),
                "top1_freq": float(channel_top1_count[channel_pos, expert_idx] / denom_channel),
                "contribution_evaluated": False,
                "channel_rmse_delta_if_removed": 0.0,
                "channel_rmse_delta_ratio_if_removed": 0.0,
                "channel_r2_delta_if_removed": 0.0,
                "channel_contribution_score": 0.0,
            }
            channel_expert_by_key[(channel_pos, expert_idx)] = row
            channel_expert_rows.append(row)

    summary = {
        "num_windows": int(total_windows),
        "num_route_slots": int(total_route_slots),
        "num_alive_experts": int(model.num_alive_experts()),
        "final_weight_entropy": _distribution_entropy(avg_final),
        "top1_entropy": _distribution_entropy(top1_freq),
        "max_avg_final_weight": float(avg_final.max()),
        "min_avg_final_weight": float(avg_final.min()),
        "max_selected_freq": float(selected_freq.max()),
        "min_selected_freq": float(selected_freq.min()),
        "health_signal_type": "usage",
    }

    active_health_rows = [row for row in rows if str(row["status"]) != "hard_pruned"]
    if scheduler is not None and len(contribution_eval_indices) > 0 and len(active_health_rows) > 1:
        print(
            f"   [Health] Contribution baseline... "
            f"windows={len(contribution_eval_indices)} | experts={len(active_health_rows)}"
        )
        baseline_metrics, baseline_horizon_rows, baseline_channel_rows = _evaluate_subset_metrics(
            model,
            scheduler,
            dataset,
            device,
            contribution_eval_indices,
            target_idx=target_idx,
            pred_len=pred_len,
            horizon_ranges=horizon_ranges,
            sample_sampler=sample_sampler,
            sample_inference_steps=sample_inference_steps,
            sample_eta=sample_eta,
            sample_t_start=sample_t_start,
            strict_sample_shape=strict_sample_shape,
            use_amp=use_amp,
            drift_correction=drift_correction,
            drift_tail_window=drift_tail_window,
            drift_alpha=drift_alpha,
            sample_seed=sample_seed,
            batch_size=batch_size,
            forecast_objective=forecast_objective,
            seasonal_period=seasonal_period,
            eval_space=eval_space,
            inference_mode=inference_mode,
        )

        if baseline_metrics is not None:
            print(
                f"   [Health] Baseline ready | RMSE={float(baseline_metrics['RMSE']):.4f} "
                f"| R2={float(baseline_metrics['R2']):.4f}"
            )
            baseline_horizons = {str(row["horizon"]): row for row in baseline_horizon_rows}
            baseline_channels = {int(row["channel_pos"]): row for row in baseline_channel_rows}
            contribution_scores = []
            progress = _ProgressReporter("Health Ablation", len(active_health_rows), min_interval_sec=10.0)
            completed = 0
            for expert_idx, row in enumerate(rows):
                if str(row["status"]) == "hard_pruned":
                    continue

                with _temporary_expert_gate_scale(model, expert_idx, 0.0):
                    ablated_metrics, ablated_horizon_rows, ablated_channel_rows = _evaluate_subset_metrics(
                        model,
                        scheduler,
                        dataset,
                        device,
                        contribution_eval_indices,
                        target_idx=target_idx,
                        pred_len=pred_len,
                        horizon_ranges=horizon_ranges,
                        sample_sampler=sample_sampler,
                        sample_inference_steps=sample_inference_steps,
                        sample_eta=sample_eta,
                        sample_t_start=sample_t_start,
                        strict_sample_shape=strict_sample_shape,
                        use_amp=use_amp,
                        drift_correction=drift_correction,
                        drift_tail_window=drift_tail_window,
                        drift_alpha=drift_alpha,
                        sample_seed=sample_seed,
                        batch_size=batch_size,
                        forecast_objective=forecast_objective,
                        seasonal_period=seasonal_period,
                        eval_space=eval_space,
                        inference_mode=inference_mode,
                    )

                if ablated_metrics is None:
                    continue

                base_rmse = float(baseline_metrics["RMSE"])
                rmse_delta = float(ablated_metrics["RMSE"] - base_rmse)
                rmse_delta_ratio = float(rmse_delta / max(abs(base_rmse), 1e-8))
                r2_delta = float(float(baseline_metrics["R2"]) - float(ablated_metrics["R2"]))

                ablated_horizons = {str(item["horizon"]): item for item in ablated_horizon_rows}
                ablated_channels = {int(item["channel_pos"]): item for item in ablated_channel_rows}
                horizon_details = {}
                best_label = ""
                best_delta = 0.0
                best_ratio = float("-inf")
                worst_label = ""
                worst_delta = 0.0
                worst_ratio = float("inf")

                for label, _, _ in horizon_ranges:
                    label = str(label)
                    base_row = baseline_horizons.get(label)
                    ablated_row = ablated_horizons.get(label)
                    if base_row is None or ablated_row is None:
                        continue

                    base_h_rmse = float(base_row["RMSE"])
                    delta = float(float(ablated_row["RMSE"]) - base_h_rmse)
                    ratio = float(delta / max(abs(base_h_rmse), 1e-8))
                    horizon_details[label] = {
                        "rmse_delta": delta,
                        "rmse_delta_ratio": ratio,
                    }

                    if ratio > best_ratio:
                        best_label = label
                        best_delta = delta
                        best_ratio = ratio
                    if ratio < worst_ratio:
                        worst_label = label
                        worst_delta = delta
                        worst_ratio = ratio

                if best_label == "":
                    best_delta = rmse_delta
                    best_ratio = rmse_delta_ratio
                if worst_label == "":
                    worst_delta = rmse_delta
                    worst_ratio = rmse_delta_ratio

                contribution_score = float(max(rmse_delta_ratio, best_ratio))
                contribution_scores.append(contribution_score)

                row["contribution_evaluated"] = True
                row["rmse_delta_if_removed"] = rmse_delta
                row["rmse_delta_ratio_if_removed"] = rmse_delta_ratio
                row["r2_delta_if_removed"] = r2_delta
                row["best_horizon_if_removed"] = best_label
                row["best_horizon_rmse_delta_if_removed"] = float(best_delta)
                row["best_horizon_rmse_delta_ratio_if_removed"] = float(best_ratio)
                row["worst_horizon_if_removed"] = worst_label
                row["worst_horizon_rmse_delta_if_removed"] = float(worst_delta)
                row["worst_horizon_rmse_delta_ratio_if_removed"] = float(worst_ratio)
                row["contribution_score"] = contribution_score
                row["horizon_rmse_delta_if_removed_json"] = json.dumps(
                    horizon_details,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for channel_pos, base_ch_row in baseline_channels.items():
                    ablated_ch_row = ablated_channels.get(int(channel_pos))
                    channel_expert_row = channel_expert_by_key.get((int(channel_pos), int(expert_idx)))
                    if ablated_ch_row is None or channel_expert_row is None:
                        continue
                    base_ch_rmse = float(base_ch_row["RMSE"])
                    ch_rmse_delta = float(float(ablated_ch_row["RMSE"]) - base_ch_rmse)
                    ch_ratio = float(ch_rmse_delta / max(abs(base_ch_rmse), 1e-8))
                    ch_r2_delta = float(float(base_ch_row["R2"]) - float(ablated_ch_row["R2"]))
                    channel_expert_row["contribution_evaluated"] = True
                    channel_expert_row["channel_rmse_delta_if_removed"] = ch_rmse_delta
                    channel_expert_row["channel_rmse_delta_ratio_if_removed"] = ch_ratio
                    channel_expert_row["channel_r2_delta_if_removed"] = ch_r2_delta
                    channel_expert_row["channel_contribution_score"] = ch_ratio
                completed += 1
                progress.update(
                    completed,
                    extra=(
                        f"expert={row['expert_name']} | "
                        f"delta_rmse={rmse_delta:+.4f} | "
                        f"best_h={best_label or '-'}"
                    ),
                )

            summary.update(
                {
                    "health_signal_type": "marginal_contribution",
                    "contribution_num_windows": int(len(contribution_eval_indices)),
                    "contribution_baseline_rmse": float(baseline_metrics["RMSE"]),
                    "contribution_baseline_r2": float(baseline_metrics["R2"]),
                    "contribution_evaluated_experts": int(sum(1 for row in rows if row["contribution_evaluated"])),
                }
            )
            if contribution_scores:
                summary["max_contribution_score"] = float(max(contribution_scores))
                summary["min_contribution_score"] = float(min(contribution_scores))
    channel_expert_summary_rows = []
    for channel_pos in range(channel_final_sum.shape[0]):
        rows_for_channel = [
            row for row in channel_expert_rows
            if int(row["channel_pos"]) == int(channel_pos) and str(row["status"]) != "hard_pruned"
        ]
        if not rows_for_channel:
            continue
        by_selected = max(rows_for_channel, key=lambda row: float(row["selected_freq"]))
        by_dense = max(rows_for_channel, key=lambda row: float(row["avg_final_dense_weight"]))
        by_contrib = max(rows_for_channel, key=lambda row: float(row["channel_contribution_score"]))
        channel_name = str(target_names[channel_pos]) if channel_pos < len(target_names) else f"ch{channel_pos}"
        channel_idx = int(target_idx[channel_pos]) if channel_pos < len(target_idx) else int(channel_pos)
        channel_expert_summary_rows.append(
            {
                "channel_idx": channel_idx,
                "channel_pos": int(channel_pos),
                "channel_name": channel_name,
                "top_selected_expert_idx": int(by_selected["expert_idx"]),
                "top_selected_expert_name": str(by_selected["expert_name"]),
                "top_selected_freq": float(by_selected["selected_freq"]),
                "top_dense_expert_idx": int(by_dense["expert_idx"]),
                "top_dense_expert_name": str(by_dense["expert_name"]),
                "top_dense_weight": float(by_dense["avg_final_dense_weight"]),
                "top_contribution_expert_idx": int(by_contrib["expert_idx"]),
                "top_contribution_expert_name": str(by_contrib["expert_name"]),
                "top_contribution_score": float(by_contrib["channel_contribution_score"]),
            }
        )
    return rows, summary, channel_expert_rows, channel_expert_summary_rows


def maybe_update_expert_lifecycle(model, health_rows, epoch, phase, config, save_dir, channel_expert_rows=None):
    if not isinstance(model, MoEDiffuser):
        return None
    if not bool(config.get("enable_expert_lifecycle", True)):
        return None

    epoch = int(epoch)
    start_epoch = int(config.get("expert_retire_start_epoch", 5))
    check_every = max(1, int(config.get("expert_retire_check_every", 1)))
    if epoch < start_epoch or ((epoch - start_epoch) % check_every != 0):
        return None

    by_idx = {int(row["expert_idx"]): row for row in (health_rows or [])}
    if not by_idx:
        return None

    soft_usage_threshold = float(config.get("soft_retire_usage_threshold", 0.02))
    hard_usage_threshold = float(config.get("hard_prune_usage_threshold", 0.005))
    soft_contribution_threshold = float(config.get("soft_retire_contribution_threshold", 0.0))
    hard_contribution_threshold = float(config.get("hard_prune_contribution_threshold", 0.0))
    soft_patience = int(config.get("soft_retire_patience", 2))
    hard_patience = int(config.get("hard_prune_patience", 4))
    protect_channel_specialists = bool(config.get("protect_channel_specialists", True))
    specialist_selected_threshold = float(config.get("channel_specialist_selected_threshold", 0.5))
    specialist_contribution_threshold = float(config.get("channel_specialist_contribution_threshold", 0.0))
    protected_experts = set()
    if protect_channel_specialists:
        for ch_row in channel_expert_rows or []:
            expert_idx = int(ch_row.get("expert_idx", -1))
            if expert_idx < 0:
                continue
            selected = float(ch_row.get("selected_freq", 0.0))
            contribution = float(ch_row.get("channel_contribution_score", 0.0))
            contribution_ready = bool(ch_row.get("contribution_evaluated", False))
            if selected >= specialist_selected_threshold:
                protected_experts.add(expert_idx)
            if contribution_ready and contribution > specialist_contribution_threshold:
                protected_experts.add(expert_idx)

    for expert_idx in range(model.num_experts):
        row = by_idx.get(expert_idx)
        if row is None:
            continue
        status = str(row["status"])
        if status == "hard_pruned":
            continue

        health_signal, signal_type = _resolve_expert_health_signal(row)
        threshold = (
            hard_contribution_threshold if status == "soft_retired" else soft_contribution_threshold
        ) if signal_type == "marginal_contribution" else (
            hard_usage_threshold if status == "soft_retired" else soft_usage_threshold
        )
        row["health_signal"] = float(health_signal)
        row["health_signal_type"] = str(signal_type)
        if health_signal < threshold:
            model.increment_expert_bad_epoch(expert_idx)
            row["bad_epochs"] = int(model.expert_bad_epochs[expert_idx].item())
        else:
            model.reset_expert_bad_epoch(expert_idx)
            row["bad_epochs"] = 0

    action = None

    hard_candidates = []
    for row in by_idx.values():
        if str(row["status"]) != "soft_retired":
            continue
        expert_idx = int(row["expert_idx"])
        if expert_idx in protected_experts:
            continue
        if int(row["bad_epochs"]) < hard_patience:
            continue
        health_signal, signal_type = _resolve_expert_health_signal(row)
        threshold = hard_contribution_threshold if signal_type == "marginal_contribution" else hard_usage_threshold
        if health_signal >= threshold:
            continue
        hard_candidates.append((health_signal, -int(row["bad_epochs"]), expert_idx))

    if hard_candidates and model.can_prune_more():
        _, _, target_idx = sorted(hard_candidates)[0]
        if model.hard_prune_expert(target_idx, epoch=epoch):
            row = by_idx[target_idx]
            health_signal, signal_type = _resolve_expert_health_signal(row)
            action = {
                "phase": str(phase),
                "epoch": int(epoch),
                "action": "hard_prune",
                "expert_idx": int(target_idx),
                "expert_name": str(row["expert_name"]),
                "health_signal": float(health_signal),
                "health_signal_type": str(signal_type),
                "selected_freq": float(row["selected_freq"]),
                "avg_final_weight": float(row["avg_final_weight"]),
                "rmse_delta_ratio_if_removed": float(row.get("rmse_delta_ratio_if_removed", 0.0)),
                "best_horizon_rmse_delta_ratio_if_removed": float(
                    row.get("best_horizon_rmse_delta_ratio_if_removed", 0.0)
                ),
                "bad_epochs": int(model.expert_bad_epochs[target_idx].item()),
                "alive_experts_after": int(model.num_alive_experts()),
            }

    if action is None:
        soft_candidates = []
        for row in by_idx.values():
            if str(row["status"]) != "active":
                continue
            expert_idx = int(row["expert_idx"])
            if expert_idx in protected_experts:
                continue
            if int(row["bad_epochs"]) < soft_patience:
                continue
            health_signal, signal_type = _resolve_expert_health_signal(row)
            threshold = soft_contribution_threshold if signal_type == "marginal_contribution" else soft_usage_threshold
            if health_signal >= threshold:
                continue
            soft_candidates.append((health_signal, -int(row["bad_epochs"]), expert_idx))

        if soft_candidates:
            _, _, target_idx = sorted(soft_candidates)[0]
            if model.soft_retire_expert(target_idx, epoch=epoch):
                row = by_idx[target_idx]
                health_signal, signal_type = _resolve_expert_health_signal(row)
                action = {
                    "phase": str(phase),
                    "epoch": int(epoch),
                    "action": "soft_retire",
                    "expert_idx": int(target_idx),
                    "expert_name": str(row["expert_name"]),
                    "health_signal": float(health_signal),
                    "health_signal_type": str(signal_type),
                    "selected_freq": float(row["selected_freq"]),
                    "avg_final_weight": float(row["avg_final_weight"]),
                    "rmse_delta_ratio_if_removed": float(row.get("rmse_delta_ratio_if_removed", 0.0)),
                    "best_horizon_rmse_delta_ratio_if_removed": float(
                        row.get("best_horizon_rmse_delta_ratio_if_removed", 0.0)
                    ),
                    "bad_epochs": int(model.expert_bad_epochs[target_idx].item()),
                    "alive_experts_after": int(model.num_alive_experts()),
                }

    if action is not None:
        lifecycle_path = os.path.join(save_dir, "expert_lifecycle.jsonl")
        _append_jsonl(lifecycle_path, action)
        print(
            f"   [Lifecycle] {action['action']} -> "
            f"{action['expert_name']} (idx={action['expert_idx']})"
        )
    elif protected_experts:
        protected_names = [model.expert_names[i] for i in sorted(protected_experts) if 0 <= i < model.num_experts]
        print(f"   [Lifecycle] protected channel specialists: {', '.join(protected_names)}")
    return action
