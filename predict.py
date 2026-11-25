import os
import contextlib
import json
import joblib
import inspect
import sys
import argparse
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
import matplotlib.pyplot as plt

from utils.scheduler import DDPMScheduler
from utils.metrics import MetricsCalculator, resolve_horizon_ranges, compute_overall_and_per_channel
from utils.postprocess import apply_linear_drift_correction
from utils.common import resolve_context_pred_lengths, resolve_label_path
from utils.config import validate_config
from datasets.generic_csv import GenericCSVDataset
from training.core import get_model_instance
from training.baselines import (
    normalize_forecast_objective,
    restore_residual_forecast_bcl,
    uses_residual_forecast_objective,
)
from network.moe import MoEDiffuser


def set_global_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_lc_from_cl(x_cl, name="x"):
    x = _to_numpy(x_cl)
    if x.ndim == 1:
        return x.reshape(-1, 1)

    if x.ndim != 2:
        raise RuntimeError(f"{name} expects 2D, got {x.shape}")

    if x.shape[0] <= 64 and x.shape[1] > x.shape[0]:
        return x.T
    return x


def _bcl_to_lc(x_bcl, name="x_bcl"):
    x = _to_numpy(x_bcl)
    if x.ndim != 3:
        raise RuntimeError(f"{name} expects [B,C,L], got {x.shape}")
    return x[0].transpose(1, 0)


def _restore_generated_forecast(gen_bcl, past_bcl, forecast_objective, seasonal_period):
    """Undo the residual diffusion target so predictions live in load units.

    With `daily_seasonal_residual` the denoiser models `future - repeat(past tail)`.
    Sampling therefore yields a residual trajectory; it must be added back to the
    seasonal baseline *before* inverse scaling, otherwise metrics compare a
    residual against raw loads (and the numbers are meaningless).
    """
    objective = normalize_forecast_objective(forecast_objective)
    if uses_residual_forecast_objective(objective):
        return restore_residual_forecast_bcl(past_bcl, gen_bcl, period=int(seasonal_period))
    return gen_bcl


def _infer_target_indices(feature_cols, target_cols):
    if not feature_cols or not target_cols:
        return None
    idx = []
    f2i = {c: i for i, c in enumerate(feature_cols)}
    for tc in target_cols:
        if tc in f2i:
            idx.append(int(f2i[tc]))
    return idx if len(idx) > 0 else None


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_dataset_with_train_assets(
    seq_len,
    context_len,
    pred_len,
    csv_path,
    mode,
    scaler=None,
    feature_cols=None,
    target_cols=None,
    time_col=None,
    channels_first=True,
    add_time_features=False,
    add_cyc_features=False,
    label_path=None,
    target_only_diffusion=False,
    split_borders=None,
    split_ratios=None,
    scaler_type="minmax",
):
    sig = inspect.signature(GenericCSVDataset.__init__)
    params = sig.parameters

    kwargs = {
        "seq_len": int(seq_len),
        "csv_file_path": csv_path,
        "mode": mode,
    }
    if "context_len" in params:
        kwargs["context_len"] = int(context_len)
    if "pred_len" in params:
        kwargs["pred_len"] = int(pred_len)

    if "channels_first" in params:
        kwargs["channels_first"] = bool(channels_first)
    if "add_time_features" in params:
        kwargs["add_time_features"] = bool(add_time_features)
    if "add_cyc_features" in params:
        kwargs["add_cyc_features"] = bool(add_cyc_features)
    if "target_only_diffusion" in params:
        kwargs["target_only_diffusion"] = bool(target_only_diffusion)
    if time_col is not None and "time_col" in params:
        kwargs["time_col"] = time_col

    if scaler is not None and "scaler" in params:
        kwargs["scaler"] = scaler
    if feature_cols is not None and "feature_cols" in params:
        kwargs["feature_cols"] = feature_cols
    if target_cols is not None and "target_cols" in params:
        kwargs["target_cols"] = target_cols
    if label_path is not None and "label_path" in params:
        kwargs["label_path"] = label_path
    if split_borders is not None and "split_borders" in params:
        kwargs["split_borders"] = split_borders
    if split_ratios is not None and "split_ratios" in params:
        kwargs["split_ratios"] = split_ratios
    if scaler_type is not None and "scaler_type" in params:
        kwargs["scaler_type"] = scaler_type

    ds = GenericCSVDataset(**kwargs)

    if scaler is not None and "scaler" not in params:
        print("⚠️ [Predict] GenericCSVDataset 不支持 __init__ 传入 scaler。")

    return ds


def unpack_dataset_item(item, device: torch.device):
    if isinstance(item, dict):
        past = item["past"]
        future = item.get("future", None)
        known_future = item.get("known_future", None)
        coarse_label = item.get("coarse_label", None)
    elif isinstance(item, (list, tuple)) and len(item) == 3:
        past, future, coarse_label = item
        known_future = None
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        past, future = item
        coarse_label = None
        known_future = None
    else:
        raise RuntimeError(f"Unexpected dataset item format: {type(item)}")

    past = past.to(device)
    if future is not None:
        future = future.to(device)
    if known_future is not None:
        known_future = known_future.to(device)
    if coarse_label is not None:
        coarse_label = coarse_label.to(device).long()
        if coarse_label.dim() == 0:
            coarse_label = coarse_label.view(1)
    return past, future, coarse_label, known_future


def _route_weights_for_condition(model, past_bcl, coarse_label_b=None, known_future_b=None):
    if not isinstance(model, MoEDiffuser):
        return None
    with torch.no_grad():
        _, weight_dict = model(
            None,
            None,
            past_bcl,
            coarse_label=coarse_label_b,
            known_future=known_future_b,
            return_weights=True,
        )
    return weight_dict


def _print_route_selection(route_weights, expert_names, target_names=None, prefix="Route"):
    if route_weights is None:
        return
    if isinstance(route_weights, dict):
        final_weights = route_weights.get("final", None)
        dense_weights = route_weights.get("final_dense", final_weights)
    else:
        final_weights = route_weights
        dense_weights = route_weights
    if final_weights is None:
        return
    weights = final_weights.detach().cpu()
    dense = dense_weights.detach().cpu() if isinstance(dense_weights, torch.Tensor) else weights
    if weights.dim() != 3 or weights.size(0) <= 0:
        return
    selected = torch.argmax(weights[0], dim=-1).tolist()
    target_names = list(target_names or [])
    print(f"   [{prefix}] channel -> expert")
    for task_i, expert_idx in enumerate(selected):
        channel_name = target_names[task_i] if task_i < len(target_names) else f"ch{task_i}"
        expert_name = expert_names[int(expert_idx)] if int(expert_idx) < len(expert_names) else str(expert_idx)
        selected_weight = float(weights[0, task_i, int(expert_idx)])
        dense_weight = float(dense[0, task_i, int(expert_idx)])
        is_hard = bool(
            torch.isclose(weights[0, task_i].sum(), torch.tensor(1.0, dtype=weights.dtype))
            and int((weights[0, task_i] > 0).sum().item()) == 1
        )
        print(
            f"      {channel_name}: {expert_name} | "
            f"selected_weight={selected_weight:.3f} | dense_weight={dense_weight:.3f} | hard={is_hard}"
        )


def _load_channel_best_expert(experiment_folder, file_name=None):
    file_name = file_name or "channel_best_expert.json"
    path = file_name if os.path.isabs(file_name) else os.path.join(experiment_folder, file_name)
    if not os.path.exists(path):
        if file_name != "channel_best_expert.json":
            print(f"[ChannelBest] file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"⚠️ channel_best_expert.json 读取失败: {exc}")
        return None
    channels = payload.get("channels", [])
    if not isinstance(channels, list):
        print(f"[ChannelBest] invalid channels payload: {path}")
        return None
    payload["_source_path"] = path
    return payload


def _print_channel_best_summary(payload, prefix="ChannelBest", max_rows=None):
    if not payload:
        return
    channels = payload.get("channels", [])
    if not channels:
        return
    source = payload.get("_source_path", "")
    print(f"   [{prefix}] channel best expert summary" + (f" | source={source}" if source else ""))
    limit = len(channels) if max_rows is None else min(int(max_rows), len(channels))
    for row in channels[:limit]:
        channel_name = str(row.get("channel_name", row.get("channel_idx", "")))
        selected = str(row.get("top_selected_expert_name", ""))
        selected_freq = float(row.get("top_selected_freq", 0.0) or 0.0)
        prior = str(row.get("prior_expert_name", ""))
        prior_weight = float(row.get("prior_weight", 0.0) or 0.0)
        print(
            f"      {channel_name}: selected={selected} ({selected_freq:.3f}) | "
            f"prior={prior} ({prior_weight:.3f})"
        )


def predict(experiment_folder, channel_best_file=None):
    config_path = os.path.join(experiment_folder, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config, mode="predict")
    set_global_seed(int(config.get("seed", 42)))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_len, pred_len = resolve_context_pred_lengths(config)
    config["context_len"] = int(context_len)
    config["pred_len"] = int(pred_len)
    config["seq_len"] = int(pred_len)
    model_name = config["model"]
    csv_path = config.get("csv_path", None)
    add_time_features = bool(config.get("add_time_features", False))
    add_cyc_features = bool(config.get("add_cyc_features", False))
    drift_correction = bool(config.get("drift_correction", True))
    drift_tail_window = int(config.get("drift_tail_window", 24))
    drift_alpha = float(config.get("drift_alpha", 0.75))
    use_amp = bool(config.get("use_amp", True)) and DEVICE.type == "cuda"
    predict_sampler = str(
        config.get(
            "predict_sampler",
            config.get("final_eval_sampler", config.get("eval_sampler", "ddim")),
        )
    )
    predict_inference_steps = int(
        config.get(
            "predict_inference_steps",
            config.get("final_eval_inference_steps", config.get("eval_inference_steps", 100)),
        )
    )
    predict_eta = float(
        config.get(
            "predict_eta",
            config.get("final_eval_eta", config.get("eval_eta", 0.0)),
        )
    )
    predict_t_start = config.get("ddim_t_start", None)
    sample_seed_base = int(config.get("seed", 42)) + 2001
    forecast_objective = normalize_forecast_objective(config.get("forecast_objective", "direct"))
    seasonal_period = int(max(1, config.get("seasonal_period", 24)))
    inference_mode = str(config.get("inference_mode", "diffusion")).strip().lower()

    print(f"📂 已加载实验配置: {experiment_folder}")
    print(
        f"   Model={model_name} | context_len={context_len} | pred_len={pred_len} | "
        f"sampler={predict_sampler} | steps={predict_inference_steps} | "
        f"objective={forecast_objective}(period={seasonal_period})"
    )

    scaler_path = os.path.join(experiment_folder, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print("❌ 找不到 scaler.pkl，无法还原数据尺度。")
        return
    scaler = joblib.load(scaler_path)

    feature_cols = None
    feature_cols_path = os.path.join(experiment_folder, "feature_cols.json")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

    target_cols = None
    target_cols_path = os.path.join(experiment_folder, "target_cols.json")
    if os.path.exists(target_cols_path):
        with open(target_cols_path, "r", encoding="utf-8") as f:
            target_cols = json.load(f)

    inferred_target_indices = _infer_target_indices(feature_cols, target_cols)
    if inferred_target_indices is not None:
        print(f"🎯 [Predict] inferred target_indices={inferred_target_indices} (from target_cols.json)")
    else:
        print("⚠️ [Predict] 未能从 target_cols.json 推断 target_indices，将尝试从 dataset.target_indices 获取。")

    if not csv_path or not os.path.exists(csv_path):
        print(f"⚠️ config.csv_path 不存在或已移动: {csv_path}")
        print("   你仍可用模式 [2] 手动指定新 CSV 推理。")

    test_dataset = None
    if csv_path and os.path.exists(csv_path):
        test_label_path = resolve_label_path(config, csv_path, "test")
        test_dataset = build_dataset_with_train_assets(
            seq_len=pred_len,
            context_len=context_len,
            pred_len=pred_len,
            csv_path=csv_path,
            mode="test",
            scaler=scaler,
            feature_cols=feature_cols,
            target_cols=target_cols,
            time_col=config.get("time_col", None),
            channels_first=True,
            add_time_features=add_time_features,
            add_cyc_features=add_cyc_features,
            label_path=test_label_path,
            target_only_diffusion=bool(config.get("target_only_diffusion", False)),
            split_borders=config.get("split_borders", None),
            split_ratios=config.get("split_ratios", None),
            scaler_type=str(config.get("scaler_type", "minmax")),
        )
        if len(test_dataset) == 0:
            print("⚠️ 测试集样本不足（len=0）")
            test_dataset = None

    if test_dataset is not None:
        NUM_CHANNELS = int(test_dataset.num_channels)
        target_indices = getattr(test_dataset, "target_indices", None) or inferred_target_indices
        known_future_channels = int(getattr(test_dataset, "known_future_channels", 0))
    else:
        if feature_cols is not None:
            if bool(config.get("target_only_diffusion", False)) and target_cols:
                NUM_CHANNELS = len(target_cols)
                known_future_channels = max(0, len(feature_cols) - len(target_cols))
            else:
                NUM_CHANNELS = len(feature_cols)
                known_future_channels = 0
        else:
            print("❌ 无法确定 num_channels。")
            return
        target_indices = inferred_target_indices

    if test_dataset is None and bool(config.get("target_only_diffusion", False)) and target_cols:
        target_indices = list(range(len(target_cols)))

    if not target_indices:
        target_indices = list(range(NUM_CHANNELS))
        print("⚠️ [Predict] target_indices 为空，已降级为全部通道作为任务（不推荐）。")

    print(f"✅ Channels(C)={NUM_CHANNELS} | Tasks(T)={len(target_indices)}")
    route_target_names = list(target_cols or [])
    if len(route_target_names) != len(target_indices):
        route_target_names = [f"ch{idx}" for idx in target_indices]
    channel_best_payload = _load_channel_best_expert(experiment_folder, file_name=channel_best_file)
    _print_channel_best_summary(channel_best_payload, prefix="LoadedChannelBest")

    model = get_model_instance(
        config,
        NUM_CHANNELS,
        DEVICE,
        target_indices=target_indices,
        known_future_channels=known_future_channels,
    )

    weights_path = os.path.join(experiment_folder, "model_weights.pth")
    if not os.path.exists(weights_path):
        print("❌ 找不到模型权重: model_weights.pth")
        return

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    scheduler = DDPMScheduler(
        num_timesteps=1000,
        device=DEVICE,
        schedule=config.get("noise_schedule", "linear"),
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )

    while True:
        print("\n" + "=" * 60)
        print(
            f" 🚀 推理模式 | Model={model_name} | C={NUM_CHANNELS} | "
            f"T={len(target_indices)} | Context={context_len} | Pred={pred_len}"
        )
        print("=" * 60)
        print(" [1] 🏆 在测试集上评估 (Test Set Evaluation)")
        print(" [2] 🔮 预测新的 CSV 文件 (Predict New File)")
        print(" [q] 🚪 退出 (Quit)")

        choice = input("\n👉 请选择模式: ").strip().lower()
        if choice == "q":
            break

        if choice == "1":
            if test_dataset is None:
                print("❌ 当前没有可用测试集")
                continue

            num_samples = min(len(test_dataset), 50)
            print(f"📊 测试集样本数: {len(test_dataset)}，评估前 {num_samples} 条（只算任务通道）...")

            maes, mses, rmses = [], [], []
            per_mae_acc = None
            per_mse_acc = None
            per_rmse_acc = None
            horizon_ranges = resolve_horizon_ranges(pred_len, boundaries=config.get("horizon_boundaries", None))
            horizon_true = {label: [] for label, _, _ in horizon_ranges}
            horizon_pred = {label: [] for label, _, _ in horizon_ranges}
            route_counts = None

            for i in range(num_samples):
                past_cl, true_future_cl, coarse_label, known_future = unpack_dataset_item(test_dataset[i], DEVICE)
                coarse_label_b = coarse_label.unsqueeze(0) if coarse_label is not None else None
                known_future_b = known_future.unsqueeze(0) if known_future is not None else None

                past_bcl = past_cl.unsqueeze(0)
                route_weights = _route_weights_for_condition(
                    model,
                    past_bcl,
                    coarse_label_b=coarse_label_b,
                    known_future_b=known_future_b,
                )
                if route_weights is not None:
                    final_route_weights = (
                        route_weights.get("final", None) if isinstance(route_weights, dict) else route_weights
                    )
                    if final_route_weights is None:
                        continue
                    selected = torch.argmax(final_route_weights.detach().cpu(), dim=-1)
                    if route_counts is None:
                        route_counts = np.zeros((selected.shape[1], len(model.expert_names)), dtype=np.int64)
                    for task_i, expert_idx in enumerate(selected[0].tolist()):
                        route_counts[int(task_i), int(expert_idx)] += 1

                with torch.no_grad(), make_autocast_context(DEVICE, use_amp):
                    if inference_mode == "direct":
                        pred_bcl = model(
                            torch.zeros_like(past_bcl),
                            torch.zeros(past_bcl.size(0), dtype=torch.long, device=DEVICE),
                            past_bcl,
                            coarse_label=coarse_label_b,
                            known_future=known_future_b,
                            return_weights=False,
                        )
                        if isinstance(pred_bcl, tuple):
                            pred_bcl = pred_bcl[0]
                    else:
                        pred_bcl = scheduler.sample(
                        model,
                        past_bcl,
                        pred_len,
                        coarse_label=coarse_label_b,
                        known_future=known_future_b,
                        seed=sample_seed_base + int(i),
                        sampler=predict_sampler,
                        num_inference_steps=predict_inference_steps,
                        eta=predict_eta,
                        t_start=predict_t_start,
                        strict_seq_len=bool(config.get("strict_sample_shape", True)),
                    )
                pred_bcl = _restore_generated_forecast(
                    pred_bcl,
                    past_bcl,
                    forecast_objective,
                    seasonal_period,
                )

                true_lc = _ensure_lc_from_cl(true_future_cl, "true_future")
                pred_lc = _bcl_to_lc(pred_bcl, "pred_bcl")

                if hasattr(test_dataset, "inverse_transform"):
                    true_raw = test_dataset.inverse_transform(true_lc)
                    pred_raw = test_dataset.inverse_transform(pred_lc)
                else:
                    true_raw, pred_raw = true_lc, pred_lc

                if drift_correction:
                    past_lc = _ensure_lc_from_cl(past_cl, "past")
                    if hasattr(test_dataset, "inverse_transform"):
                        past_raw = test_dataset.inverse_transform(past_lc)
                    else:
                        past_raw = past_lc
                    pred_raw = apply_linear_drift_correction(
                        past_raw,
                        pred_raw,
                        target_indices=target_indices,
                        tail_window=drift_tail_window,
                        alpha=drift_alpha,
                    )

                mae, mse, rmse, per_mae, per_mse, per_rmse = compute_overall_and_per_channel(
                    true_raw,
                    pred_raw,
                    target_idx=target_indices,
                )
                maes.append(mae)
                mses.append(mse)
                rmses.append(rmse)
                true_eval = np.asarray(true_raw)[:, target_indices]
                pred_eval = np.asarray(pred_raw)[:, target_indices]
                for label, start_h, end_h in horizon_ranges:
                    horizon_true[label].extend(true_eval[start_h:end_h, :].reshape(-1).tolist())
                    horizon_pred[label].extend(pred_eval[start_h:end_h, :].reshape(-1).tolist())

                if per_mae_acc is None:
                    per_mae_acc = np.zeros(len(per_mae), dtype=float)
                    per_mse_acc = np.zeros(len(per_mse), dtype=float)
                    per_rmse_acc = np.zeros(len(per_rmse), dtype=float)
                per_mae_acc += np.array(per_mae)
                per_mse_acc += np.array(per_mse)
                per_rmse_acc += np.array(per_rmse)

                if i % 10 == 0:
                    print(f"   Sample {i}: MAE={mae:.4f} MSE={mse:.4f} RMSE={rmse:.4f}")

            print("\n" + "-" * 40)
            print(f"✅ 测试集评估报告 (Top {num_samples}) - 仅任务通道")
            print(f"   Overall MAE : {np.mean(maes):.4f}")
            print(f"   Overall MSE : {np.mean(mses):.4f}")
            print(f"   Overall RMSE: {np.mean(rmses):.4f}")

            per_mae_mean = per_mae_acc / num_samples
            per_mse_mean = per_mse_acc / num_samples
            per_rmse_mean = per_rmse_acc / num_samples
            print("   Per-Task-Channel MAE :", np.round(per_mae_mean, 4).tolist())
            print("   Per-Task-Channel MSE :", np.round(per_mse_mean, 4).tolist())
            print("   Per-Task-Channel RMSE:", np.round(per_rmse_mean, 4).tolist())
            for label, _, _ in horizon_ranges:
                if len(horizon_true[label]) == 0:
                    continue
                h_metrics = MetricsCalculator.compute(horizon_true[label], horizon_pred[label])
                print(
                    f"   Horizon {label} RMSE: {float(h_metrics['RMSE']):.4f} "
                    f"| MSE: {float(h_metrics['MSE']):.4f} "
                    f"| MAE: {float(h_metrics['MAE']):.4f}"
                )
            if route_counts is not None:
                print("   Channel expert selection frequency:")
                denom = float(max(1, num_samples))
                for task_i in range(route_counts.shape[0]):
                    channel_name = route_target_names[task_i] if task_i < len(route_target_names) else f"ch{task_i}"
                    expert_idx = int(np.argmax(route_counts[task_i]))
                    expert_name = model.expert_names[expert_idx]
                    freq = float(route_counts[task_i, expert_idx] / denom)
                    print(f"      {channel_name}: {expert_name} ({freq:.3f})")
            _print_channel_best_summary(channel_best_payload, prefix="TrainedChannelBest")
            print("-" * 40)

        elif choice == "2":
            new_csv = input("📂 请输入 CSV 文件路径: ").strip().strip('"')
            if not os.path.exists(new_csv):
                print("❌ 文件不存在。")
                continue

            try:
                tmp_ds = build_dataset_with_train_assets(
                    seq_len=pred_len,
                    context_len=context_len,
                    pred_len=pred_len,
                    csv_path=new_csv,
                    mode="test",
                    scaler=scaler,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    channels_first=True,
                    add_time_features=add_time_features,
                    add_cyc_features=add_cyc_features,
                    label_path=None,  # 新 CSV 默认没有预分析标签，自动回退到 learned coarse
                    target_only_diffusion=bool(config.get("target_only_diffusion", False)),
            split_borders=config.get("split_borders", None),
            split_ratios=config.get("split_ratios", None),
            scaler_type=str(config.get("scaler_type", "minmax")),
                )
                if len(tmp_ds) == 0:
                    print("⚠️ 文件有效样本不足（len=0）")
                    continue

                past_cl, _, coarse_label, known_future = unpack_dataset_item(tmp_ds[len(tmp_ds) - 1], DEVICE)
                coarse_label_b = coarse_label.unsqueeze(0) if coarse_label is not None else None
                known_future_b = known_future.unsqueeze(0) if known_future is not None else None

                past_bcl = past_cl.unsqueeze(0)
                route_weights = _route_weights_for_condition(
                    model,
                    past_bcl,
                    coarse_label_b=coarse_label_b,
                    known_future_b=known_future_b,
                )
                _print_route_selection(
                    route_weights,
                    getattr(model, "expert_names", []),
                    target_names=route_target_names,
                    prefix="PredictRoute",
                )
                _print_channel_best_summary(channel_best_payload, prefix="TrainedChannelBest")

                with torch.no_grad(), make_autocast_context(DEVICE, use_amp):
                    if inference_mode == "direct":
                        pred_bcl = model(
                            torch.zeros_like(past_bcl),
                            torch.zeros(past_bcl.size(0), dtype=torch.long, device=DEVICE),
                            past_bcl,
                            coarse_label=coarse_label_b,
                            known_future=known_future_b,
                            return_weights=False,
                        )
                        if isinstance(pred_bcl, tuple):
                            pred_bcl = pred_bcl[0]
                    else:
                        pred_bcl = scheduler.sample(
                        model,
                        past_bcl,
                        pred_len,
                        coarse_label=coarse_label_b,
                        known_future=known_future_b,
                        seed=sample_seed_base + int(len(tmp_ds) - 1),
                        sampler=predict_sampler,
                        num_inference_steps=predict_inference_steps,
                        eta=predict_eta,
                        t_start=predict_t_start,
                        strict_seq_len=bool(config.get("strict_sample_shape", True)),
                    )
                pred_bcl = _restore_generated_forecast(
                    pred_bcl,
                    past_bcl,
                    forecast_objective,
                    seasonal_period,
                )

                past_lc = _ensure_lc_from_cl(past_cl, "past")
                pred_lc = _bcl_to_lc(pred_bcl, "pred_bcl")

                if hasattr(tmp_ds, "inverse_transform"):
                    past_raw = tmp_ds.inverse_transform(past_lc)
                    pred_raw = tmp_ds.inverse_transform(pred_lc)
                else:
                    past_raw, pred_raw = past_lc, pred_lc

                if drift_correction:
                    pred_raw = apply_linear_drift_correction(
                        past_raw,
                        pred_raw,
                        target_indices=target_indices,
                        tail_window=drift_tail_window,
                        alpha=drift_alpha,
                    )

                draw_ch = target_indices[-1] if target_indices else (pred_raw.shape[1] - 1)

                hist = np.asarray(past_raw)[:, draw_ch]
                fut = np.asarray(pred_raw)[:, draw_ch]

                plt.figure(figsize=(10, 5))
                plt.plot(range(context_len), hist, label="History", linewidth=2)
                plt.plot(
                    range(context_len, context_len + pred_len),
                    fut,
                    label="Forecast",
                    linestyle="--",
                    linewidth=2,
                )
                plt.title(f"Forecast: {os.path.basename(new_csv)} | draw_channel={draw_ch}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                save_img = os.path.join(experiment_folder, "latest_forecast.png")
                plt.savefig(save_img)
                plt.close()
                print(f"📉 预测图已保存: {save_img}")

            except Exception as e:
                print(f"❌ 推理出错: {e}")
                import traceback
                traceback.print_exc()

        else:
            print("⚠️ 无效选择。")


def _resolve_channel_best_file(args):
    if args.channel_best_file:
        return args.channel_best_file
    if args.channel_best_epoch:
        safe = []
        for ch in str(args.channel_best_epoch).strip():
            safe.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
        tag = "".join(safe).strip("_")
        if tag:
            return f"channel_best_expert_{tag}.json"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive prediction/evaluation for a saved experiment.")
    parser.add_argument("--experiment_folder", default=None, help="Saved experiment folder. If omitted, choose from results/.")
    parser.add_argument("--channel_best_file", default=None, help="Optional channel_best_expert JSON path or file name.")
    parser.add_argument("--channel_best_epoch", default=None, help="Load channel_best_expert_<epoch>.json from the experiment folder.")
    args = parser.parse_args()

    channel_best_file = _resolve_channel_best_file(args)
    if args.experiment_folder:
        predict(os.path.abspath(args.experiment_folder), channel_best_file=channel_best_file)
        raise SystemExit

    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ 找不到 {results_dir}，请先训练。")
        raise SystemExit

    experiments = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
    if not experiments:
        print("❌ results 下没有实验记录。")
        raise SystemExit

    print(f"\n🔎 发现 {len(experiments)} 个实验记录:")
    for i, exp in enumerate(experiments):
        print(f" [{i+1}] {exp}")

    idx_str = input("\n👉 请输入实验序号 (ID): ").strip()
    try:
        idx = int(idx_str) - 1
    except ValueError:
        print("❌ 输入无效。")
        raise SystemExit

    if not (0 <= idx < len(experiments)):
        print("❌ ID 越界。")
        raise SystemExit

    target_exp = os.path.join(results_dir, experiments[idx])
    predict(target_exp, channel_best_file=channel_best_file)
