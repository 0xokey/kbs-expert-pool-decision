import math

import numpy as np
import torch


DEFAULT_BASELINE_METHODS = (
    "persistence",
    "daily_seasonal",
    "weekly_seasonal",
    "history_mean",
)


def normalize_forecast_objective(value) -> str:
    objective = str(value or "direct").strip().lower()
    aliases = {
        "none": "direct",
        "raw": "direct",
        "absolute": "direct",
        "seasonal_residual": "daily_seasonal_residual",
        "daily_residual": "daily_seasonal_residual",
    }
    objective = aliases.get(objective, objective)
    if objective not in {"direct", "daily_seasonal_residual"}:
        raise ValueError(
            "[Config] forecast_objective must be 'direct' or "
            f"'daily_seasonal_residual', got {value!r}"
        )
    return objective


def uses_residual_forecast_objective(value) -> bool:
    return normalize_forecast_objective(value) == "daily_seasonal_residual"


def make_repeated_seasonal_baseline_bcl(
    past_bcl: torch.Tensor,
    pred_len: int,
    period: int = 24,
) -> torch.Tensor:
    if past_bcl.dim() != 3:
        raise RuntimeError(f"past_bcl should be [B,C,L], got {tuple(past_bcl.shape)}")
    pred_len = int(pred_len)
    if pred_len <= 0:
        raise ValueError(f"pred_len must be positive, got {pred_len}")
    history_len = int(past_bcl.shape[-1])
    if history_len <= 0:
        raise ValueError("past_bcl history length must be positive")

    period = int(max(1, min(int(period), history_len)))
    pattern = past_bcl[:, :, -period:]
    repeats = int(math.ceil(pred_len / period))
    return pattern.repeat(1, 1, repeats)[:, :, :pred_len].contiguous()


def make_residual_target_bcl(
    past_bcl: torch.Tensor,
    future_bcl: torch.Tensor,
    period: int = 24,
) -> torch.Tensor:
    baseline = make_repeated_seasonal_baseline_bcl(
        past_bcl,
        pred_len=int(future_bcl.shape[-1]),
        period=period,
    )
    if baseline.shape != future_bcl.shape:
        raise RuntimeError(
            f"residual baseline shape {tuple(baseline.shape)} != future shape {tuple(future_bcl.shape)}"
        )
    return future_bcl - baseline


def restore_residual_forecast_bcl(
    past_bcl: torch.Tensor,
    generated_residual_bcl: torch.Tensor,
    period: int = 24,
) -> torch.Tensor:
    baseline = make_repeated_seasonal_baseline_bcl(
        past_bcl,
        pred_len=int(generated_residual_bcl.shape[-1]),
        period=period,
    )
    if baseline.shape != generated_residual_bcl.shape:
        raise RuntimeError(
            f"residual baseline shape {tuple(baseline.shape)} != generated shape "
            f"{tuple(generated_residual_bcl.shape)}"
        )
    return generated_residual_bcl + baseline


def _as_lc_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise RuntimeError(f"baseline input should be [L,C], got {arr.shape}")
    return arr


def _repeat_tail_np(past_lc, pred_len: int, period: int) -> np.ndarray:
    past_lc = _as_lc_array(past_lc)
    pred_len = int(pred_len)
    if pred_len <= 0:
        raise ValueError(f"pred_len must be positive, got {pred_len}")
    if past_lc.shape[0] <= 0:
        raise ValueError("past_lc history length must be positive")

    period = int(max(1, min(int(period), past_lc.shape[0])))
    pattern = past_lc[-period:, :]
    repeats = int(math.ceil(pred_len / period))
    return np.tile(pattern, (repeats, 1))[:pred_len, :]


def make_baseline_prediction_np(
    method: str,
    past_lc,
    pred_len: int,
    seasonal_period: int = 24,
) -> np.ndarray:
    method = str(method).strip().lower()
    past_lc = _as_lc_array(past_lc)
    pred_len = int(pred_len)

    if method == "persistence":
        return np.repeat(past_lc[-1:, :], pred_len, axis=0)
    if method == "daily_seasonal":
        return _repeat_tail_np(past_lc, pred_len=pred_len, period=seasonal_period)
    if method == "weekly_seasonal":
        return _repeat_tail_np(past_lc, pred_len=pred_len, period=pred_len)
    if method == "history_mean":
        return np.repeat(np.mean(past_lc, axis=0, keepdims=True), pred_len, axis=0)

    raise ValueError(f"Unknown baseline method: {method!r}")


def metric_skill(model_metrics: dict, baseline_metrics: dict) -> dict:
    def rel_gain(key: str):
        model_value = float(model_metrics[key])
        baseline_value = float(baseline_metrics[key])
        return float(1.0 - model_value / max(abs(baseline_value), 1e-12))

    return {
        "MAE_skill": rel_gain("MAE"),
        "MSE_skill": rel_gain("MSE"),
        "RMSE_skill": rel_gain("RMSE"),
        "sMAPE_skill": rel_gain("sMAPE"),
        "R2_delta": float(float(model_metrics["R2"]) - float(baseline_metrics["R2"])),
    }
