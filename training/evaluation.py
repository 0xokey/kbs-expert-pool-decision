import json
import os

import numpy as np
import torch

from utils.visualizer import save_prediction_plot, save_series_overview_plot
from utils.metrics import MetricsCalculator, resolve_horizon_ranges
from utils.postprocess import apply_linear_drift_correction
from network.moe import MoEDiffuser
from training.core import unpack_sample
from training.common import (
    _ProgressReporter,
    _bcl_to_blc,
    _bcl_to_lc,
    _get_target_names,
    _jsonable_metrics,
    _limit_eval_indices,
    _safe_tag,
    _stringify_time_value,
    _write_csv_rows,
    make_autocast_context,
    select_eval_indices,
)
from training.baselines import (
    DEFAULT_BASELINE_METHODS,
    make_baseline_prediction_np,
    metric_skill,
    normalize_forecast_objective,
    restore_residual_forecast_bcl,
    uses_residual_forecast_objective,
)
from training.health import collect_moe_expert_health


def run_evaluation(
    model,
    scheduler,
    dataset,
    epoch_name,
    device,
    save_dir,
    export_all_windows=False,
    max_eval_windows=None,
    index_seed: int = 0,
    sample_seed: int | None = None,
    save_preview_plot: bool = True,
    save_window_plots: bool = True,
    window_plot_limit: int | None = None,
    save_overview_plots: bool = True,
    sample_sampler: str = "ddpm",
    sample_inference_steps: int | None = None,
    sample_eta: float = 0.0,
    sample_t_start: int | None = None,
    use_amp: bool = False,
    drift_correction: bool = False,
    drift_tail_window: int = 24,
    drift_alpha: float = 0.75,
    horizon_boundaries=None,
    save_expert_health: bool = False,
    expert_health_max_windows: int | None = None,
    crps_samples: int = 0,
    crps_num_windows: int | None = None,
    eval_batch_size: int = 1,
    strict_sample_shape: bool = True,
    forecast_objective: str = "direct",
    seasonal_period: int = 24,
    baseline_methods=None,
    baseline_reference: str = "daily_seasonal",
    eval_space: str = "raw",
    eval_ensemble_samples: int = 1,
    inference_mode: str = "diffusion",
):
    model.eval()

    if len(dataset) == 0:
        print(f"   [Eval {epoch_name}] dataset is empty, skip.")
        model.train()
        return None

    if export_all_windows:
        indices = list(range(len(dataset)))
    else:
        if max_eval_windows is None:
            max_eval_windows = min(5, len(dataset))
        indices = select_eval_indices(len(dataset), max_eval_windows, seed=index_seed)

    target_idx = getattr(dataset, "target_indices", None)
    if not target_idx:
        target_idx = list(range(getattr(dataset, "num_channels", 1)))
    target_names = _get_target_names(dataset, target_idx)
    context_len = int(getattr(dataset, "context_len", getattr(dataset, "seq_len", 0)))
    pred_len = int(getattr(dataset, "pred_len", getattr(dataset, "seq_len", 0)))
    horizon_ranges = resolve_horizon_ranges(pred_len, boundaries=horizon_boundaries)
    forecast_objective = normalize_forecast_objective(forecast_objective)
    seasonal_period = int(max(1, seasonal_period))
    residual_objective = uses_residual_forecast_objective(forecast_objective)
    baseline_methods = list(baseline_methods or DEFAULT_BASELINE_METHODS)
    baseline_reference = str(baseline_reference or "daily_seasonal").strip().lower()
    eval_space = str(eval_space or "raw").strip().lower()
    if eval_space not in {"raw", "scaled"}:
        raise ValueError(f"eval_space must be 'raw' or 'scaled', got {eval_space!r}")
    use_scaled_space = eval_space == "scaled"
    eval_ensemble_samples = int(max(1, eval_ensemble_samples))
    inference_mode = str(inference_mode or "diffusion").strip().lower()
    if inference_mode not in {"diffusion", "direct"}:
        raise ValueError(f"inference_mode must be 'diffusion' or 'direct', got {inference_mode!r}")

    preds_list, trues_list = [], []
    sample_rows = []
    channel_expert_selection_rows = []
    per_channel_true = [[] for _ in target_idx]
    per_channel_pred = [[] for _ in target_idx]
    horizon_true = {label: [] for label, _, _ in horizon_ranges}
    horizon_pred = {label: [] for label, _, _ in horizon_ranges}
    baseline_store = {}
    for method in baseline_methods:
        method = str(method).strip().lower()
        baseline_store[method] = {
            "true": [],
            "pred": [],
            "per_channel_true": [[] for _ in target_idx],
            "per_channel_pred": [[] for _ in target_idx],
            "horizon_true": {label: [] for label, _, _ in horizon_ranges},
            "horizon_pred": {label: [] for label, _, _ in horizon_ranges},
        }
    baseline_sample_rows = []

    eval_dir = None
    plots_dir = None
    preview_dir = None
    overview_dir = None
    metrics_dir = None
    overview_time = None
    overview_true = None
    overview_sum = None
    overview_count = None
    plotted_window_indices = set()

    if export_all_windows:
        eval_dir = os.path.join(save_dir, "evaluation", _safe_tag(epoch_name))
        overview_dir = os.path.join(eval_dir, "plots", "overview")
        metrics_dir = os.path.join(eval_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        if save_window_plots:
            plots_dir = os.path.join(eval_dir, "plots", "windows")
            os.makedirs(plots_dir, exist_ok=True)
            limit = None if window_plot_limit is None else int(window_plot_limit)
            if limit is not None and limit > 0:
                plotted_window_indices = set(select_eval_indices(len(dataset), limit, seed=index_seed + 17))
            elif limit is None:
                plotted_window_indices = set(indices)
        if save_overview_plots:
            os.makedirs(overview_dir, exist_ok=True)

        if hasattr(dataset, "data_tensor"):
            slice_data = dataset.data_tensor.detach().cpu().numpy()
            slice_raw = dataset.inverse_transform(slice_data) if hasattr(dataset, "inverse_transform") else slice_data
            overview_true = np.asarray(slice_raw)[:, target_idx]
            overview_sum = np.zeros_like(overview_true, dtype=np.float64)
            overview_count = np.zeros_like(overview_true, dtype=np.int32)

        if hasattr(dataset, "time_slice") and dataset.time_slice is not None:
            overview_time = np.asarray(dataset.time_slice)
        elif overview_true is not None:
            overview_time = np.arange(len(overview_true))
    else:
        eval_dir = os.path.join(save_dir, "evaluation", _safe_tag(epoch_name))
        metrics_dir = os.path.join(eval_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        if save_preview_plot:
            preview_dir = os.path.join(eval_dir, "plots", "preview")
            os.makedirs(preview_dir, exist_ok=True)

    eval_batch_size = int(max(1, eval_batch_size))
    active_inference_steps = scheduler.num_timesteps if sample_inference_steps is None else int(sample_inference_steps)
    print(
        f"   Evaluating ({epoch_name})... windows={len(indices)} | "
        f"sampler={sample_sampler} | steps={active_inference_steps} | batch={eval_batch_size} | "
        f"objective={forecast_objective}"
    )
    eval_progress = _ProgressReporter(f"Eval {epoch_name}", len(indices), min_interval_sec=30.0)

    processed = 0
    for batch_start in range(0, len(indices), eval_batch_size):
        batch_indices = [int(v) for v in indices[batch_start: batch_start + eval_batch_size]]
        batch_items = [unpack_sample(dataset[idx], device) for idx in batch_indices]

        past_bcl = torch.stack([item[0] for item in batch_items], dim=0)
        true_bcl = torch.stack([item[1] for item in batch_items], dim=0)

        coarse_items = [item[2] for item in batch_items]
        if any(label is not None for label in coarse_items):
            coarse_label_b = torch.stack(
                [
                    label if label is not None else torch.zeros_like(next(v for v in coarse_items if v is not None))
                    for label in coarse_items
                ],
                dim=0,
            )
        else:
            coarse_label_b = None

        known_items = [item[3] for item in batch_items]
        if any(known is not None and known.numel() > 0 for known in known_items):
            template = next(v for v in known_items if v is not None and v.numel() > 0)
            # Missing known covariates are padded as zero by contract; KnownFutureAdapter is zero-initialized.
            known_future_b = torch.stack(
                [known if known is not None else torch.zeros_like(template) for known in known_items],
                dim=0,
            )
        else:
            known_future_b = None

        route_final_b = None
        route_final_dense_b = None
        if isinstance(model, MoEDiffuser):
            with torch.no_grad(), make_autocast_context(device, use_amp):
                _, route_weight_dict = model(
                    None,
                    None,
                    past_bcl,
                    coarse_label=coarse_label_b,
                    known_future=known_future_b,
                    return_weights=True,
                )
            route_final_b = route_weight_dict["final"].detach().cpu().numpy()
            route_final_dense_b = route_weight_dict.get("final_dense", route_weight_dict["final"]).detach().cpu().numpy()

        with torch.no_grad(), make_autocast_context(device, use_amp):
            sample_seed_i = None if sample_seed is None else int(sample_seed) + int(batch_indices[0])
            gen_bcl = None
            if inference_mode == "direct":
                # Deterministic experts: a single forward pass, no diffusion
                # trajectory. Used for the regression variant of the framework.
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
            for ensemble_i in range(0 if inference_mode == "direct" else eval_ensemble_samples):
                seed_i = None if sample_seed_i is None else sample_seed_i + ensemble_i * 104729
                sample_bcl = scheduler.sample(
                    model,
                    past_bcl,
                    pred_len,
                    coarse_label=coarse_label_b,
                    known_future=known_future_b,
                    seed=seed_i,
                    sampler=sample_sampler,
                    num_inference_steps=sample_inference_steps,
                    eta=sample_eta,
                    t_start=sample_t_start,
                    strict_seq_len=strict_sample_shape,
                )
                gen_bcl = sample_bcl if gen_bcl is None else gen_bcl + sample_bcl
            if inference_mode != "direct":
                gen_bcl = gen_bcl / float(eval_ensemble_samples)
            if residual_objective:
                gen_bcl = restore_residual_forecast_bcl(
                    past_bcl,
                    gen_bcl,
                    period=seasonal_period,
                )

        past_blc = _bcl_to_blc(past_bcl)
        true_blc = _bcl_to_blc(true_bcl)
        pred_blc = _bcl_to_blc(gen_bcl)

        for local_i, idx in enumerate(batch_indices):
            past_lc = past_blc[local_i]
            true_lc = true_blc[local_i]
            pred_lc = pred_blc[local_i]

            if use_scaled_space:
                # TSLib-style reporting: metrics on standardized values so numbers
                # are comparable with the published forecasting literature.
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

            if hasattr(dataset, "get_time_window"):
                hist_time, fut_time = dataset.get_time_window(idx)
            else:
                hist_time = np.arange(context_len)
                fut_time = np.arange(context_len, context_len + pred_len)
            plot_dir_rel = ""

            true_t = np.asarray(true_raw)[:, target_idx]
            pred_t = np.asarray(pred_raw)[:, target_idx]
            trues_list.extend(true_t.reshape(-1).tolist())
            preds_list.extend(pred_t.reshape(-1).tolist())

            if route_final_b is not None:
                route_final = route_final_b[local_i]
                route_final_dense = route_final_dense_b[local_i]
                for channel_pos, channel_idx in enumerate(target_idx):
                    selected_expert_idx = int(np.argmax(route_final[channel_pos]))
                    selected_expert_name = (
                        model.expert_names[selected_expert_idx]
                        if selected_expert_idx < len(model.expert_names)
                        else str(selected_expert_idx)
                    )
                    channel_expert_selection_rows.append(
                        {
                            "sample_idx": int(idx),
                            "channel_idx": int(channel_idx),
                            "channel_pos": int(channel_pos),
                            "channel_name": str(target_names[channel_pos]),
                            "selected_expert_idx": selected_expert_idx,
                            "selected_expert_name": selected_expert_name,
                            "selected_weight": float(route_final[channel_pos, selected_expert_idx]),
                            "selected_dense_weight": float(route_final_dense[channel_pos, selected_expert_idx]),
                            "is_hard_one_hot": bool(
                                np.isclose(route_final[channel_pos].sum(), 1.0)
                                and np.count_nonzero(route_final[channel_pos] > 0) == 1
                            ),
                        }
                    )

            sample_metrics = _jsonable_metrics(MetricsCalculator.compute(true_t.reshape(-1), pred_t.reshape(-1)))
            past_t = np.asarray(past_raw)[:, target_idx]
            for baseline_method, store in baseline_store.items():
                baseline_pred_t = make_baseline_prediction_np(
                    baseline_method,
                    past_t,
                    pred_len=pred_len,
                    seasonal_period=seasonal_period,
                )
                baseline_metrics = _jsonable_metrics(
                    MetricsCalculator.compute(true_t.reshape(-1), baseline_pred_t.reshape(-1))
                )
                store["true"].extend(true_t.reshape(-1).tolist())
                store["pred"].extend(baseline_pred_t.reshape(-1).tolist())
                for channel_pos in range(len(target_idx)):
                    store["per_channel_true"][channel_pos].extend(true_t[:, channel_pos].reshape(-1).tolist())
                    store["per_channel_pred"][channel_pos].extend(
                        baseline_pred_t[:, channel_pos].reshape(-1).tolist()
                    )
                for label, start_h, end_h in horizon_ranges:
                    store["horizon_true"][label].extend(true_t[start_h:end_h, :].reshape(-1).tolist())
                    store["horizon_pred"][label].extend(
                        baseline_pred_t[start_h:end_h, :].reshape(-1).tolist()
                    )
                baseline_sample_rows.append({
                    "sample_idx": int(idx),
                    "baseline": baseline_method,
                    "history_start": _stringify_time_value(hist_time[0]),
                    "history_end": _stringify_time_value(hist_time[-1]),
                    "future_start": _stringify_time_value(fut_time[0]),
                    "future_end": _stringify_time_value(fut_time[-1]),
                    "MAE": baseline_metrics["MAE"],
                    "MSE": baseline_metrics["MSE"],
                    "RMSE": baseline_metrics["RMSE"],
                    "MAPE": baseline_metrics["MAPE"],
                    "MSPE": baseline_metrics["MSPE"],
                    "sMAPE": baseline_metrics["sMAPE"],
                    "R2": baseline_metrics["R2"],
                })
            for channel_pos in range(len(target_idx)):
                per_channel_true[channel_pos].extend(true_t[:, channel_pos].reshape(-1).tolist())
                per_channel_pred[channel_pos].extend(pred_t[:, channel_pos].reshape(-1).tolist())
            for label, start_h, end_h in horizon_ranges:
                horizon_true[label].extend(true_t[start_h:end_h, :].reshape(-1).tolist())
                horizon_pred[label].extend(pred_t[start_h:end_h, :].reshape(-1).tolist())

            if export_all_windows:
                start = int(idx) + context_len
                end = start + pred_len
                if overview_sum is not None and overview_count is not None:
                    overview_sum[start:end, :] += pred_t
                    overview_count[start:end, :] += 1

                if plots_dir is not None and (len(plotted_window_indices) == 0 or int(idx) in plotted_window_indices):
                    sample_dir = os.path.join(plots_dir, f"sample_{int(idx):05d}")
                    save_prediction_plot(
                        past_raw, true_raw, pred_raw,
                        f"{epoch_name}_idx{int(idx):05d}", sample_dir,
                        dataset=dataset,
                        sample_idx=idx,
                        use_target_only=True,
                        max_channels=None,
                    )
                    plot_dir_rel = os.path.relpath(sample_dir, eval_dir)
            elif save_preview_plot and processed == 0 and preview_dir is not None:
                save_prediction_plot(
                    past_raw, true_raw, pred_raw,
                    epoch_name, preview_dir,
                    dataset=dataset,
                    sample_idx=idx,
                    use_target_only=True,
                    max_channels=None,
                )
                plot_dir_rel = os.path.relpath(preview_dir, eval_dir)

            sample_rows.append({
                "sample_idx": int(idx),
                "history_start": _stringify_time_value(hist_time[0]),
                "history_end": _stringify_time_value(hist_time[-1]),
                "future_start": _stringify_time_value(fut_time[0]),
                "future_end": _stringify_time_value(fut_time[-1]),
                "MAE": sample_metrics["MAE"],
                "MSE": sample_metrics["MSE"],
                "RMSE": sample_metrics["RMSE"],
                "MAPE": sample_metrics["MAPE"],
                "MSPE": sample_metrics["MSPE"],
                "sMAPE": sample_metrics["sMAPE"],
                "R2": sample_metrics["R2"],
                "plot_dir": plot_dir_rel,
            })
            processed += 1
            eval_progress.update(
                processed,
                extra=f"last_idx={int(idx)} | last_rmse={float(sample_metrics['RMSE']):.4f}",
            )

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

    baseline_summary_rows = []
    baseline_channel_rows = []
    baseline_horizon_rows = []
    baseline_metric_map = {}
    baseline_channel_metric_map = {}
    baseline_horizon_metric_map = {}
    for baseline_method, store in baseline_store.items():
        if len(store["true"]) == 0:
            continue
        b_metrics = _jsonable_metrics(MetricsCalculator.compute(
            np.nan_to_num(np.asarray(store["true"], dtype=np.float64), nan=0.0),
            np.nan_to_num(np.asarray(store["pred"], dtype=np.float64), nan=0.0),
        ))
        baseline_metric_map[baseline_method] = b_metrics
        baseline_summary_rows.append({
            "baseline": baseline_method,
            "num_windows": int(len(indices)),
            "seasonal_period": int(seasonal_period),
            "MAE": b_metrics["MAE"],
            "MSE": b_metrics["MSE"],
            "RMSE": b_metrics["RMSE"],
            "MAPE": b_metrics["MAPE"],
            "MSPE": b_metrics["MSPE"],
            "sMAPE": b_metrics["sMAPE"],
            "R2": b_metrics["R2"],
        })

        baseline_channel_metric_map[baseline_method] = {}
        for channel_idx, channel_name, y_true, y_pred in zip(
            target_idx,
            target_names,
            store["per_channel_true"],
            store["per_channel_pred"],
        ):
            ch_metrics = _jsonable_metrics(MetricsCalculator.compute(y_true, y_pred))
            baseline_channel_metric_map[baseline_method][str(channel_name)] = ch_metrics
            baseline_channel_rows.append({
                "baseline": baseline_method,
                "channel_idx": int(channel_idx),
                "channel_name": channel_name,
                "MAE": ch_metrics["MAE"],
                "MSE": ch_metrics["MSE"],
                "RMSE": ch_metrics["RMSE"],
                "MAPE": ch_metrics["MAPE"],
                "MSPE": ch_metrics["MSPE"],
                "sMAPE": ch_metrics["sMAPE"],
                "R2": ch_metrics["R2"],
            })

        baseline_horizon_metric_map[baseline_method] = {}
        for label, start_h, end_h in horizon_ranges:
            if len(store["horizon_true"][label]) == 0:
                continue
            h_metrics = _jsonable_metrics(MetricsCalculator.compute(
                np.nan_to_num(np.asarray(store["horizon_true"][label], dtype=np.float64), nan=0.0),
                np.nan_to_num(np.asarray(store["horizon_pred"][label], dtype=np.float64), nan=0.0),
            ))
            baseline_horizon_metric_map[baseline_method][str(label)] = h_metrics
            baseline_horizon_rows.append({
                "baseline": baseline_method,
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

    baseline_skill_payload = {
        "reference": baseline_reference,
        "positive_skill_means_model_better": True,
        "seasonal_period": int(seasonal_period),
        "summary": {},
        "by_channel": {},
        "by_horizon": {},
    }
    for baseline_method, b_metrics in baseline_metric_map.items():
        baseline_skill_payload["summary"][baseline_method] = metric_skill(metrics, b_metrics)
        baseline_skill_payload["by_channel"][baseline_method] = {}
        for row in baseline_channel_rows:
            if row["baseline"] != baseline_method:
                continue
            model_row = None
            # Channel rows are built later for export, so compute model channel skill here directly.
            channel_pos = list(target_names).index(row["channel_name"]) if row["channel_name"] in target_names else None
            if channel_pos is not None:
                model_row = _jsonable_metrics(MetricsCalculator.compute(
                    per_channel_true[channel_pos],
                    per_channel_pred[channel_pos],
                ))
            if model_row is not None:
                baseline_skill_payload["by_channel"][baseline_method][str(row["channel_name"])] = metric_skill(
                    model_row,
                    row,
                )
        baseline_skill_payload["by_horizon"][baseline_method] = {}
        model_horizon_map = {str(row["horizon"]): row for row in horizon_rows}
        for row in baseline_horizon_rows:
            if row["baseline"] != baseline_method:
                continue
            model_row = model_horizon_map.get(str(row["horizon"]))
            if model_row is not None:
                baseline_skill_payload["by_horizon"][baseline_method][str(row["horizon"])] = metric_skill(
                    model_row,
                    row,
                )

    expert_health_rows = None
    expert_health_summary = None
    channel_expert_health_rows = None
    channel_expert_summary_rows = None
    health_contribution_indices = []
    if save_expert_health and isinstance(model, MoEDiffuser):
        health_contribution_indices = _limit_eval_indices(indices, expert_health_max_windows, seed=index_seed + 97)
        print(
            f"   [Health] Routing stats windows={len(indices)} | "
            f"contribution windows={len(health_contribution_indices)}"
        )
        (
            expert_health_rows,
            expert_health_summary,
            channel_expert_health_rows,
            channel_expert_summary_rows,
        ) = collect_moe_expert_health(
            model,
            dataset,
            device,
            indices=indices,
            contribution_indices=health_contribution_indices,
            batch_size=eval_batch_size,
            use_amp=use_amp,
            scheduler=scheduler,
            sample_sampler=sample_sampler,
            sample_inference_steps=sample_inference_steps,
            sample_eta=sample_eta,
            sample_t_start=sample_t_start,
            strict_sample_shape=strict_sample_shape,
            sample_seed=sample_seed,
            drift_correction=drift_correction,
            drift_tail_window=drift_tail_window,
            drift_alpha=drift_alpha,
            target_idx=target_idx,
            pred_len=pred_len,
            horizon_ranges=horizon_ranges,
            forecast_objective=forecast_objective,
            seasonal_period=seasonal_period,
            eval_space=eval_space,
        )

    # CRPS: multi-sample ensemble evaluation (diffusion-native metric).
    # Keep the model in eval mode so stochastic layers do not affect scoring.
    crps_samples = int(crps_samples)
    crps_windows = 16 if crps_num_windows is None else int(max(1, crps_num_windows))
    crps_indices = []
    if inference_mode == "direct":
        crps_samples = 0
    if crps_samples > 1:
        crps_indices = _limit_eval_indices(indices, crps_windows, seed=index_seed + 77)
        print(f"   [CRPS] Sampling {crps_samples} trajectories on {len(crps_indices)} windows...")
        model.eval()
        crps_ensemble_by_window = {int(idx): [] for idx in crps_indices}
        crps_true_by_window = {}
        batch_size = int(max(1, min(eval_batch_size, len(crps_indices))))
        for sample_i in range(crps_samples):
            for batch_start in range(0, len(crps_indices), batch_size):
                batch_idx = [int(v) for v in crps_indices[batch_start: batch_start + batch_size]]
                items = [unpack_sample(dataset[i], device) for i in batch_idx]
                past_bcl = torch.stack([item[0] for item in items], dim=0)
                true_bcl = torch.stack([item[1] for item in items], dim=0)
                coarse_items = [item[2] for item in items]
                if any(label is not None for label in coarse_items):
                    template = next(v for v in coarse_items if v is not None)
                    coarse_label_b = torch.stack(
                        [label if label is not None else torch.zeros_like(template) for label in coarse_items],
                        dim=0,
                    )
                else:
                    coarse_label_b = None
                known_items = [item[3] for item in items]
                if any(known is not None and known.numel() > 0 for known in known_items):
                    template = next(v for v in known_items if v is not None and v.numel() > 0)
                    known_future_b = torch.stack(
                        [known if known is not None else torch.zeros_like(template) for known in known_items],
                        dim=0,
                    )
                else:
                    known_future_b = None
                with torch.no_grad(), make_autocast_context(device, use_amp):
                    gen_bcl = scheduler.sample(
                        model,
                        past_bcl,
                        pred_len,
                        coarse_label=coarse_label_b,
                        known_future=known_future_b,
                        sampler=sample_sampler,
                        num_inference_steps=sample_inference_steps,
                        eta=sample_eta,
                        t_start=sample_t_start,
                        strict_seq_len=strict_sample_shape,
                        seed=None if sample_seed is None else int(sample_seed) + sample_i * 7919 + batch_start,
                    )
                    if residual_objective:
                        gen_bcl = restore_residual_forecast_bcl(
                            past_bcl,
                            gen_bcl,
                            period=seasonal_period,
                        )
                pred_blc = _bcl_to_blc(gen_bcl)
                true_blc = _bcl_to_blc(true_bcl)
                for local_i, idx in enumerate(batch_idx):
                    pred_lc = pred_blc[local_i]
                    true_lc = true_blc[local_i]
                    if use_scaled_space:
                        pred_raw, true_raw = pred_lc, true_lc
                    elif hasattr(dataset, "inverse_transform"):
                        pred_raw = dataset.inverse_transform(pred_lc)
                        true_raw = dataset.inverse_transform(true_lc)
                    else:
                        pred_raw, true_raw = pred_lc, true_lc
                    crps_ensemble_by_window[int(idx)].append(np.asarray(pred_raw)[:, target_idx])
                    crps_true_by_window[int(idx)] = np.asarray(true_raw)[:, target_idx]
        crps_ensemble = [
            np.stack(crps_ensemble_by_window[int(idx)], axis=0) for idx in crps_indices
        ]
        crps_true = [crps_true_by_window[int(idx)] for idx in crps_indices]
        crps_val = MetricsCalculator.crps(
            np.concatenate(crps_ensemble, axis=1),
            np.concatenate(crps_true, axis=0),
        )
        metrics["CRPS"] = float(crps_val)
        print(f"      -> CRPS: {crps_val:.4f}")
    else:
        metrics["CRPS"] = None

    print(
        f"      -> MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | "
        f"RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f} | "
        f"sMAPE: {metrics['sMAPE']:.4f} | CRPS: {metrics.get('CRPS', None)}"
    )

    if metrics_dir is not None:
        channel_rows = []
        for channel_idx, channel_name, y_true, y_pred in zip(target_idx, target_names, per_channel_true, per_channel_pred):
            channel_metrics = _jsonable_metrics(MetricsCalculator.compute(y_true, y_pred))
            channel_rows.append({
                "channel_idx": int(channel_idx),
                "channel_name": channel_name,
                "MAE": channel_metrics["MAE"],
                "MSE": channel_metrics["MSE"],
                "RMSE": channel_metrics["RMSE"],
                "MAPE": channel_metrics["MAPE"],
                "MSPE": channel_metrics["MSPE"],
                "sMAPE": channel_metrics["sMAPE"],
                "R2": channel_metrics["R2"],
            })

        contribution_scope = None
        if save_expert_health and isinstance(model, MoEDiffuser):
            contribution_scope = (
                "same_as_eval"
                if [int(v) for v in health_contribution_indices] == [int(v) for v in indices]
                else "subset"
            )
        if expert_health_summary is not None:
            expert_health_summary["eval_scope"] = "full_dataset" if export_all_windows else "subset"
            expert_health_summary["eval_num_windows"] = int(len(indices))
            expert_health_summary["eval_rmse"] = float(metrics["RMSE"])
            expert_health_summary["eval_r2"] = float(metrics["R2"])
            expert_health_summary["contribution_scope"] = contribution_scope

        summary_payload = {
            "epoch": str(epoch_name),
            "num_windows": int(len(indices)),
            "scope": "full_dataset" if export_all_windows else "subset",
            "seq_len": int(pred_len),
            "context_len": int(context_len),
            "pred_len": int(pred_len),
            "num_targets": int(len(target_idx)),
            "target_names": list(target_names),
            "sampler": str(sample_sampler),
            "inference_steps": int(active_inference_steps),
            "eta": float(sample_eta),
            "t_start": None if sample_t_start is None else int(sample_t_start),
            "eval_batch_size": int(eval_batch_size),
            "index_seed": int(index_seed),
            "sample_seed": None if sample_seed is None else int(sample_seed),
            "forecast_objective": str(forecast_objective),
            "seasonal_period": int(seasonal_period),
            "eval_space": str(eval_space),
            "eval_ensemble_samples": int(eval_ensemble_samples),
            "drift_correction": bool(drift_correction),
            "drift_tail_window": int(drift_tail_window),
            "drift_alpha": float(drift_alpha),
            "baseline_reference": baseline_reference,
            "reference_baseline_RMSE": (
                None
                if baseline_reference not in baseline_metric_map
                else baseline_metric_map[baseline_reference]["RMSE"]
            ),
            "RMSE_skill_vs_reference_baseline": (
                None
                if baseline_reference not in baseline_metric_map
                else baseline_skill_payload["summary"][baseline_reference]["RMSE_skill"]
            ),
            "R2_delta_vs_reference_baseline": (
                None
                if baseline_reference not in baseline_metric_map
                else baseline_skill_payload["summary"][baseline_reference]["R2_delta"]
            ),
            "crps_samples": int(crps_samples),
            "crps_num_windows": int(len(crps_indices)),
            "expert_health_saved": bool(expert_health_rows is not None),
            "expert_health_contribution_scope": contribution_scope,
            "expert_health_contribution_num_windows": int(len(health_contribution_indices)),
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "MSPE": metrics["MSPE"],
            "sMAPE": metrics["sMAPE"],
            "R2": metrics["R2"],
            "CRPS": metrics.get("CRPS", None),
            "horizons": [row["horizon"] for row in horizon_rows],
        }

        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        summary_csv_fields = [
                "epoch",
                "num_windows",
                "scope",
                "context_len",
                "pred_len",
                "num_targets",
                "sampler",
                "inference_steps",
                "eta",
                "t_start",
                "eval_batch_size",
                "forecast_objective",
                "seasonal_period",
                "baseline_reference",
                "reference_baseline_RMSE",
                "RMSE_skill_vs_reference_baseline",
                "R2_delta_vs_reference_baseline",
                "crps_samples",
                "crps_num_windows",
                "expert_health_saved",
                "expert_health_contribution_scope",
                "expert_health_contribution_num_windows",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE",
                "MSPE",
                "sMAPE",
                "R2",
                "CRPS",
            ]
        _write_csv_rows(
            os.path.join(metrics_dir, "summary.csv"),
            [{key: summary_payload.get(key) for key in summary_csv_fields}],
            summary_csv_fields,
        )

        _write_csv_rows(
            os.path.join(metrics_dir, "channel_metrics.csv"),
            channel_rows,
            ["channel_idx", "channel_name", "MAE", "MSE", "RMSE", "MAPE", "MSPE", "sMAPE", "R2"],
        )
        _write_csv_rows(
            os.path.join(metrics_dir, "sample_metrics.csv"),
            sample_rows,
            [
                "sample_idx",
                "history_start",
                "history_end",
                "future_start",
                "future_end",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE",
                "MSPE",
                "sMAPE",
                "R2",
                "plot_dir",
            ],
        )
        if channel_expert_selection_rows:
            _write_csv_rows(
                os.path.join(metrics_dir, "channel_expert_selection.csv"),
                channel_expert_selection_rows,
                [
                    "sample_idx",
                    "channel_idx",
                    "channel_pos",
                    "channel_name",
                    "selected_expert_idx",
                    "selected_expert_name",
                    "selected_weight",
                    "selected_dense_weight",
                    "is_hard_one_hot",
                ],
            )
        if baseline_summary_rows:
            _write_csv_rows(
                os.path.join(metrics_dir, "baseline_metrics.csv"),
                baseline_summary_rows,
                ["baseline", "num_windows", "seasonal_period", "MAE", "MSE", "RMSE", "MAPE", "MSPE", "sMAPE", "R2"],
            )
            _write_csv_rows(
                os.path.join(metrics_dir, "baseline_channel_metrics.csv"),
                baseline_channel_rows,
                [
                    "baseline",
                    "channel_idx",
                    "channel_name",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "MSPE",
                    "sMAPE",
                    "R2",
                ],
            )
            _write_csv_rows(
                os.path.join(metrics_dir, "baseline_horizon_metrics.csv"),
                baseline_horizon_rows,
                [
                    "baseline",
                    "horizon",
                    "start_step",
                    "end_step",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "MSPE",
                    "sMAPE",
                    "R2",
                ],
            )
            _write_csv_rows(
                os.path.join(metrics_dir, "baseline_sample_metrics.csv"),
                baseline_sample_rows,
                [
                    "sample_idx",
                    "baseline",
                    "history_start",
                    "history_end",
                    "future_start",
                    "future_end",
                    "MAE",
                    "MSE",
                    "RMSE",
                    "MAPE",
                    "MSPE",
                    "sMAPE",
                    "R2",
                ],
            )
            with open(os.path.join(metrics_dir, "baseline_skill.json"), "w", encoding="utf-8") as f:
                json.dump(baseline_skill_payload, f, indent=2, ensure_ascii=False)
        if len(horizon_rows) > 0:
            _write_csv_rows(
                os.path.join(metrics_dir, "horizon_metrics.csv"),
                horizon_rows,
                ["horizon", "start_step", "end_step", "MAE", "MSE", "RMSE", "MAPE", "MSPE", "sMAPE", "R2"],
            )
        if expert_health_rows is not None:
            _write_csv_rows(
                os.path.join(metrics_dir, "expert_health.csv"),
                expert_health_rows,
                [
                    "expert_idx",
                    "expert_name",
                    "status",
                    "gate_scale",
                    "bad_epochs",
                    "soft_retired_epoch",
                    "hard_pruned_epoch",
                    "avg_final_weight",
                    "avg_final_dense_weight",
                    "avg_coarse_weight",
                    "avg_fine_weight",
                    "selected_freq",
                    "selection_freq",
                    "coarse_selected_freq",
                    "top1_freq",
                    "contribution_evaluated",
                    "rmse_delta_if_removed",
                    "rmse_delta_ratio_if_removed",
                    "r2_delta_if_removed",
                    "best_horizon_if_removed",
                    "best_horizon_rmse_delta_if_removed",
                    "best_horizon_rmse_delta_ratio_if_removed",
                    "worst_horizon_if_removed",
                    "worst_horizon_rmse_delta_if_removed",
                    "worst_horizon_rmse_delta_ratio_if_removed",
                    "contribution_score",
                    "horizon_rmse_delta_if_removed_json",
                ],
            )
            with open(os.path.join(metrics_dir, "expert_health_summary.json"), "w", encoding="utf-8") as f:
                json.dump(expert_health_summary or {}, f, indent=2, ensure_ascii=False)
            if channel_expert_health_rows is not None:
                _write_csv_rows(
                    os.path.join(metrics_dir, "channel_expert_health.csv"),
                    channel_expert_health_rows,
                    [
                        "channel_idx",
                        "channel_pos",
                        "channel_name",
                        "expert_idx",
                        "expert_name",
                        "status",
                        "gate_scale",
                        "avg_final_weight",
                        "avg_final_dense_weight",
                        "avg_coarse_weight",
                        "avg_fine_weight",
                        "selected_freq",
                        "selection_freq",
                        "top1_freq",
                        "contribution_evaluated",
                        "channel_rmse_delta_if_removed",
                        "channel_rmse_delta_ratio_if_removed",
                        "channel_r2_delta_if_removed",
                        "channel_contribution_score",
                    ],
                )
            if channel_expert_summary_rows is not None:
                _write_csv_rows(
                    os.path.join(metrics_dir, "channel_expert_summary.csv"),
                    channel_expert_summary_rows,
                    [
                        "channel_idx",
                        "channel_pos",
                        "channel_name",
                        "top_selected_expert_idx",
                        "top_selected_expert_name",
                        "top_selected_freq",
                        "top_dense_expert_idx",
                        "top_dense_expert_name",
                        "top_dense_weight",
                        "top_contribution_expert_idx",
                        "top_contribution_expert_name",
                        "top_contribution_score",
                    ],
                )

        manifest = {
            "summary": "summary.json",
            "summary_csv": "summary.csv",
            "channel_metrics": "channel_metrics.csv",
            "sample_metrics": "sample_metrics.csv",
            "channel_expert_selection": (
                "channel_expert_selection.csv" if channel_expert_selection_rows else None
            ),
            "baseline_metrics": "baseline_metrics.csv" if baseline_summary_rows else None,
            "baseline_channel_metrics": "baseline_channel_metrics.csv" if baseline_summary_rows else None,
            "baseline_horizon_metrics": "baseline_horizon_metrics.csv" if baseline_summary_rows else None,
            "baseline_sample_metrics": "baseline_sample_metrics.csv" if baseline_summary_rows else None,
            "baseline_skill": "baseline_skill.json" if baseline_summary_rows else None,
            "horizon_metrics": "horizon_metrics.csv" if len(horizon_rows) > 0 else None,
            "expert_health": "expert_health.csv" if expert_health_rows is not None else None,
            "expert_health_summary": "expert_health_summary.json" if expert_health_summary is not None else None,
            "channel_expert_health": (
                "channel_expert_health.csv" if channel_expert_health_rows is not None else None
            ),
            "channel_expert_summary": (
                "channel_expert_summary.csv" if channel_expert_summary_rows is not None else None
            ),
            "scope": summary_payload["scope"],
            "num_windows": summary_payload["num_windows"],
            "forecast_objective": summary_payload["forecast_objective"],
            "crps_num_windows": summary_payload["crps_num_windows"],
        }
        with open(os.path.join(metrics_dir, "metrics_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        if export_all_windows and save_overview_plots and overview_true is not None and overview_sum is not None and overview_count is not None:
            avg_pred = np.full_like(overview_true, np.nan, dtype=np.float64)
            valid_mask = overview_count > 0
            avg_pred[valid_mask] = overview_sum[valid_mask] / overview_count[valid_mask]

            for channel_pos, channel_name in enumerate(target_names):
                save_series_overview_plot(
                    overview_time,
                    overview_true[:, channel_pos],
                    avg_pred[:, channel_pos],
                    f"Validation Overview | {epoch_name} | {channel_name}",
                    os.path.join(overview_dir, f"{_safe_tag(channel_name)}.png"),
                )

        print(f"      -> Evaluation artifacts saved to: {eval_dir}")

    model.train()
    metrics["context_len"] = int(context_len)
    metrics["pred_len"] = int(pred_len)
    metrics["horizon_rows"] = horizon_rows
    metrics["baseline_rows"] = baseline_summary_rows
    metrics["baseline_skill"] = baseline_skill_payload
    metrics["expert_health_rows"] = expert_health_rows
    metrics["expert_health_summary"] = expert_health_summary
    metrics["channel_expert_health_rows"] = channel_expert_health_rows
    metrics["channel_expert_summary_rows"] = channel_expert_summary_rows
    metrics["channel_expert_selection_rows"] = channel_expert_selection_rows
    metrics["eval_dir"] = eval_dir
    return metrics
