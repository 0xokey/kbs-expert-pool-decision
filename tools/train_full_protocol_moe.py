from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]

import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.generic_csv import GenericCSVDataset
from network.full_protocol_moe import FullProtocolMoEDiffuser
from full_methods import (
    FULL_PROTOCOL_REGISTRY,
    STRICT_REPRODUCTION_STATUS,
    FullProtocolBatch,
)


AVAILABLE_STRICT_METHODS = [
    method
    for method, status in STRICT_REPRODUCTION_STATUS.items()
    if status.get("status") == "available"
]

PHASE0_SELECTION_MODE = "expert_once_multichannel_train_channelwise_select"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_int(config: dict, key: str, default: int | None = None, *, minimum: int | None = None) -> int | None:
    raw = config.get(key, default)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _as_float(config: dict, key: str, default: float | None = None, *, minimum: float | None = None) -> float | None:
    raw = config.get(key, default)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        if bool(config.get("require_cuda", False)):
            raise RuntimeError(
                "Config requested CUDA with require_cuda=true, but torch.cuda.is_available() is false. "
                "Fix the server GPU/PyTorch CUDA environment before training."
            )
        print("[Device] CUDA unavailable, falling back to CPU.")
        requested = "cpu"
    if bool(config.get("require_cuda", False)) and requested != "cuda":
        raise RuntimeError("require_cuda=true requires config.device='cuda'.")
    return torch.device(requested)


def _validate_config(config: dict, config_path: Path) -> None:
    required = ["csv_path", "context_len", "pred_len", "batch_size", "epochs", "lr"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config key(s): {missing}")

    csv_path = (ROOT / str(config["csv_path"])).resolve() if not Path(str(config["csv_path"])).is_absolute() else Path(str(config["csv_path"]))
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv_path does not exist: {csv_path} (from {config_path})")
    if csv_path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError(f"csv_path must point to a .csv or .txt table file: {csv_path}")

    _as_int(config, "context_len", minimum=1)
    _as_int(config, "pred_len", minimum=1)
    _as_int(config, "batch_size", minimum=1)
    _as_int(config, "eval_batch_size", config.get("batch_size", 1), minimum=1)
    _as_int(config, "epochs", minimum=1)
    _as_int(config, "eval_every", 5, minimum=1)
    _as_int(config, "eval_samples", 1, minimum=1)
    _as_int(config, "val_eval_max_windows", 32, minimum=1)
    _as_int(config, "test_eval_max_windows", 64, minimum=1)
    _as_int(config, "baseline_period", 24, minimum=1)
    _as_int(config, "phase0_epochs", 1, minimum=1)
    _as_int(config, "phase0_eval_samples", config.get("eval_samples", 1), minimum=1)
    _as_float(config, "lr", minimum=0.0)
    _as_float(config, "phase0_lr", config.get("lr", 1e-3), minimum=0.0)
    _as_float(config, "grad_clip_norm", None, minimum=0.0)


def _protocol_overrides(config: dict, method: str) -> dict:
    overrides = config.get("protocol_overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise TypeError("protocol_overrides must be an object when provided")
    payload = overrides.get(method, {})
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"protocol_overrides[{method!r}] must be an object")
    return dict(payload)


def _default_assignment(target_names: list[str]) -> list[str]:
    available = list(AVAILABLE_STRICT_METHODS)
    return [available[idx % len(available)] for idx, _ in enumerate(target_names)]


def _normalize_assignment(raw_assignment, target_names: list[str]) -> dict[int, str]:
    if raw_assignment is None:
        raw_assignment = _default_assignment(target_names)

    if isinstance(raw_assignment, list):
        if len(raw_assignment) != len(target_names):
            raise ValueError(
                f"channel_methods list must have {len(target_names)} entries, got {len(raw_assignment)}"
            )
        assignment = {idx: str(method) for idx, method in enumerate(raw_assignment)}
    elif isinstance(raw_assignment, dict):
        assignment = {}
        by_name = {name: idx for idx, name in enumerate(target_names)}
        for key, method in raw_assignment.items():
            if str(key).isdigit():
                idx = int(key)
            elif str(key) in by_name:
                idx = by_name[str(key)]
            else:
                raise KeyError(f"Unknown channel key {key!r}; valid names={target_names}")
            assignment[idx] = str(method)
    else:
        raise TypeError("channel_methods must be a list, dict, or null")

    expected = set(range(len(target_names)))
    if set(assignment) != expected:
        raise ValueError(f"channel_methods must cover channels {sorted(expected)}, got {sorted(assignment)}")
    unknown = sorted({method for method in assignment.values() if method not in FULL_PROTOCOL_REGISTRY})
    if unknown:
        raise KeyError(f"Unknown full protocol method(s): {unknown}; available={sorted(FULL_PROTOCOL_REGISTRY)}")
    return assignment


def _protocol_kwargs_for_assignment(config: dict, assignment: dict[int, str]) -> dict:
    grouped = defaultdict(list)
    for channel_idx, method in assignment.items():
        grouped[method].append(channel_idx)
    return {
        method: _protocol_overrides(config, method)
        for method, channels in grouped.items()
    }


def _base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _limit_dataset(dataset, max_windows: int | None):
    if max_windows is None or int(max_windows) >= len(dataset):
        return dataset
    return Subset(dataset, list(range(int(max_windows))))


def _build_dataset(config: dict, csv_path: Path, mode: str, train_dataset=None):
    train_base = _base_dataset(train_dataset) if train_dataset is not None else None
    kwargs = {
        "seq_len": int(config.get("pred_len", 24)),
        "context_len": int(config.get("context_len", 24)),
        "pred_len": int(config.get("pred_len", 24)),
        "csv_file_path": str(csv_path),
        "mode": mode,
        "channels_first": True,
        "add_time_features": bool(config.get("add_time_features", True)),
        "add_cyc_features": bool(config.get("add_cyc_features", True)),
        "target_only_diffusion": True,
    }
    if train_base is not None:
        kwargs.update(
            {
                "scaler": train_base.scaler,
                "feature_cols": train_base.feature_cols,
                "target_cols": train_base.target_cols,
                "time_col": train_base.time_col,
            }
        )
    return GenericCSVDataset(**kwargs)


def _collate_batch(batch, device: torch.device) -> FullProtocolBatch:
    if not isinstance(batch, dict):
        raise RuntimeError("Full-protocol training expects target_only_diffusion dataset dict batches")
    return FullProtocolBatch(
        past=batch["past"].to(device).float(),
        future=batch["future"].to(device).float(),
        known_future=batch.get("known_future", None).to(device).float()
        if batch.get("known_future", None) is not None
        else None,
    )


def _slice_channel_batch(batch: FullProtocolBatch, channel_idx: int) -> FullProtocolBatch:
    idx = torch.tensor([int(channel_idx)], device=batch.past.device, dtype=torch.long)
    return FullProtocolBatch(
        past=batch.past.index_select(1, idx),
        future=batch.future.index_select(1, idx),
        known_future=batch.known_future,
    )


def _bcl_to_blc(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(0, 2, 1).contiguous().numpy()


def _inverse_batch(dataset, bcl: torch.Tensor) -> np.ndarray:
    base = _base_dataset(dataset)
    arr = _bcl_to_blc(bcl)
    if not hasattr(base, "inverse_transform"):
        return arr
    return np.stack([base.inverse_transform(sample) for sample in arr], axis=0)


def _inverse_samples(dataset, bscl: torch.Tensor) -> np.ndarray:
    if bscl.dim() != 4:
        raise RuntimeError(f"Expected samples with shape [B,S,C,L], got {tuple(bscl.shape)}")
    base = _base_dataset(dataset)
    arr = bscl.detach().cpu().permute(0, 1, 3, 2).contiguous().numpy()
    if not hasattr(base, "inverse_transform"):
        return arr
    batch_size, sample_count, pred_len, channels = arr.shape
    flat = arr.reshape(batch_size * sample_count, pred_len, channels)
    raw = np.stack([base.inverse_transform(sample) for sample in flat], axis=0)
    return raw.reshape(batch_size, sample_count, pred_len, channels)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_state_dict_compat(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        message = str(exc)
        if "more than one element of the written-to tensor refers to a single memory location" not in message:
            raise
        model.load_state_dict(state, assign=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _new_metric_accumulator(num_channels: int) -> dict[str, np.ndarray]:
    return {
        "abs_sum": np.zeros(num_channels, dtype=np.float64),
        "sq_sum": np.zeros(num_channels, dtype=np.float64),
        "count": np.zeros(num_channels, dtype=np.float64),
        "crps_sum": np.zeros(num_channels, dtype=np.float64),
        "crps_count": np.zeros(num_channels, dtype=np.float64),
        "aggregate_crps_sum": np.zeros(1, dtype=np.float64),
        "aggregate_crps_count": np.zeros(1, dtype=np.float64),
    }


def _update_metric_accumulator(acc: dict[str, np.ndarray], pred_raw: np.ndarray, true_raw: np.ndarray) -> None:
    err = np.asarray(pred_raw, dtype=np.float64) - np.asarray(true_raw, dtype=np.float64)
    acc["abs_sum"] += np.abs(err).sum(axis=(0, 1))
    acc["sq_sum"] += np.square(err).sum(axis=(0, 1))
    acc["count"] += np.prod(err.shape[:2])


def _empirical_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if samples.ndim < 2:
        raise ValueError(f"CRPS samples must include a sample axis, got shape={samples.shape}")
    if samples.shape[0] < 1:
        raise ValueError("CRPS requires at least one generated sample")
    if samples.shape[1:] != truth.shape:
        raise ValueError(f"CRPS shape mismatch: samples={samples.shape}, truth={truth.shape}")

    sample_count = samples.shape[0]
    first_term = np.abs(samples - truth[None, ...]).mean(axis=0)
    sorted_samples = np.sort(samples, axis=0)
    coefficients = (2.0 * np.arange(sample_count, dtype=np.float64) - sample_count + 1.0).reshape(
        (sample_count,) + (1,) * truth.ndim
    )
    second_term = (sorted_samples * coefficients).sum(axis=0) / float(sample_count * sample_count)
    return first_term - second_term


def _update_probabilistic_metric_accumulator(
    acc: dict[str, np.ndarray],
    samples_raw: np.ndarray,
    true_raw: np.ndarray,
) -> None:
    samples = np.asarray(samples_raw, dtype=np.float64)
    truth = np.asarray(true_raw, dtype=np.float64)
    if samples.ndim != 4 or truth.ndim != 3:
        raise ValueError(f"Expected samples [B,S,L,C] and truth [B,L,C], got {samples.shape} / {truth.shape}")
    if samples.shape[0] != truth.shape[0] or samples.shape[2:] != truth.shape[1:]:
        raise ValueError(f"Probabilistic metric shape mismatch: samples={samples.shape}, truth={truth.shape}")

    # _empirical_crps expects the sample axis first.
    point_crps = _empirical_crps(np.moveaxis(samples, 1, 0), truth)
    acc["crps_sum"] += point_crps.sum(axis=(0, 1))
    acc["crps_count"] += np.prod(point_crps.shape[:2])

    aggregate_samples = samples.sum(axis=-1)
    aggregate_truth = truth.sum(axis=-1)
    aggregate_crps = _empirical_crps(np.moveaxis(aggregate_samples, 1, 0), aggregate_truth)
    acc["aggregate_crps_sum"][0] += aggregate_crps.sum()
    acc["aggregate_crps_count"][0] += aggregate_crps.size


def _finalize_metric_accumulator(acc: dict[str, np.ndarray], target_names: list[str]) -> dict:
    abs_sum = acc["abs_sum"]
    sq_sum = acc["sq_sum"]
    count = acc["count"]
    mae_ch = abs_sum / np.maximum(count, 1.0)
    mse_ch = sq_sum / np.maximum(count, 1.0)
    rmse_ch = np.sqrt(mse_ch)
    crps_ch = acc["crps_sum"] / np.maximum(acc["crps_count"], 1.0)
    crps = float(acc["crps_sum"].sum() / max(acc["crps_count"].sum(), 1.0))
    crps_sum = float(acc["aggregate_crps_sum"][0] / max(acc["aggregate_crps_count"][0], 1.0))
    return {
        "MAE": float(abs_sum.sum() / max(count.sum(), 1.0)),
        "MSE": float(sq_sum.sum() / max(count.sum(), 1.0)),
        "RMSE": float(np.sqrt(sq_sum.sum() / max(count.sum(), 1.0))),
        "CRPS": crps,
        "CRPS_sum": crps_sum,
        "channel_metrics": [
            {
                "channel": str(name),
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": float(rmse),
                "CRPS": float(crps_value),
            }
            for name, mae, mse, rmse, crps_value in zip(target_names, mae_ch, mse_ch, rmse_ch, crps_ch)
        ],
    }


def _baseline_predictions(past_raw: np.ndarray, pred_len: int, period: int) -> dict[str, np.ndarray]:
    last_value = np.repeat(past_raw[:, -1:, :], pred_len, axis=1)
    if past_raw.shape[1] >= period:
        template = past_raw[:, -period:, :]
        indices = np.arange(pred_len) % period
        daily = template[:, indices, :]
    else:
        daily = last_value.copy()
    return {
        "last_value": last_value,
        "daily_seasonal": daily,
    }


def _skill(model_mse: float, baseline_mse: float) -> float:
    if baseline_mse <= 0:
        return 0.0
    return float(1.0 - (model_mse / baseline_mse))


def _write_experiment_metadata(
    save_dir: Path,
    *,
    config: dict,
    config_path: Path,
    device: torch.device,
    csv_path: Path,
    train_dataset,
    val_dataset,
    test_dataset,
    target_names: list[str],
    assignment: dict[int, str],
    protocol_kwargs: dict,
) -> None:
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "config_path": str(config_path),
        "csv_path": str(csv_path),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data": {
            "train_windows": int(len(train_dataset)),
            "val_windows": int(len(val_dataset)),
            "test_windows": int(len(test_dataset)),
            "target_names": list(target_names),
            "context_len": int(config.get("context_len")),
            "pred_len": int(config.get("pred_len")),
        },
        "assignment": {
            str(idx): {"channel": target_names[idx], "method": method}
            for idx, method in assignment.items()
        },
        "protocol_kwargs": protocol_kwargs,
        "config": config,
    }
    with open(save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=True)


@torch.no_grad()
def _evaluate_model(
    model: FullProtocolMoEDiffuser,
    dataset,
    *,
    split_name: str,
    device: torch.device,
    batch_size: int,
    n_samples: int,
    max_windows: int | None,
    save_dir: Path,
    target_names: list[str],
    baseline_period: int,
) -> dict | None:
    if len(dataset) == 0:
        print(f"[Eval:{split_name}] dataset is empty, skipping.")
        return None

    eval_dataset = _limit_dataset(dataset, max_windows)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_acc = _new_metric_accumulator(len(target_names))
    baseline_acc = {
        "last_value": _new_metric_accumulator(len(target_names)),
        "daily_seasonal": _new_metric_accumulator(len(target_names)),
    }

    if int(n_samples) == 1:
        print(
            f"[Eval:{split_name}] n_samples=1: CRPS degenerates to absolute error. "
            "Use eval_samples > 1 for probabilistic evaluation."
        )

    model.eval()
    for batch in loader:
        protocol_batch = _collate_batch(batch, device)
        samples = model.sample(protocol_batch, n_samples=int(n_samples))
        pred_bcl = samples.mean(dim=1)
        true_bcl = protocol_batch.future

        pred_raw = _inverse_batch(eval_dataset, pred_bcl)
        true_raw = _inverse_batch(eval_dataset, true_bcl)
        past_raw = _inverse_batch(eval_dataset, protocol_batch.past)
        samples_raw = _inverse_samples(eval_dataset, samples)
        _update_metric_accumulator(model_acc, pred_raw, true_raw)
        _update_probabilistic_metric_accumulator(model_acc, samples_raw, true_raw)
        for name, baseline_raw in _baseline_predictions(
            past_raw,
            pred_len=true_raw.shape[1],
            period=int(baseline_period),
        ).items():
            _update_metric_accumulator(baseline_acc[name], baseline_raw, true_raw)
            _update_probabilistic_metric_accumulator(baseline_acc[name], baseline_raw[:, None, :, :], true_raw)

    model_metrics = _finalize_metric_accumulator(model_acc, target_names)
    baseline_metrics = {
        name: _finalize_metric_accumulator(acc, target_names)
        for name, acc in baseline_acc.items()
    }
    metrics = {
        "split": split_name,
        "windows": int(len(eval_dataset)),
        "n_samples": int(n_samples),
        "baseline_period": int(baseline_period),
        **model_metrics,
        "baselines": baseline_metrics,
        "skill": {
            name: {
                "MSE_skill": _skill(model_metrics["MSE"], baseline["MSE"]),
                "RMSE_delta": float(baseline["RMSE"] - model_metrics["RMSE"]),
            }
            for name, baseline in baseline_metrics.items()
        },
    }

    metrics_dir = save_dir / "evaluation" / split_name / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    _write_csv(
        metrics_dir / "channel_metrics.csv",
        metrics["channel_metrics"],
        ["channel", "MAE", "MSE", "RMSE", "CRPS"],
    )
    _write_csv(
        metrics_dir / "summary.csv",
        [
            {
                "split": split_name,
                "windows": metrics["windows"],
                "n_samples": metrics["n_samples"],
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "CRPS": metrics["CRPS"],
                "CRPS_sum": metrics["CRPS_sum"],
                "last_value_MSE_skill": metrics["skill"]["last_value"]["MSE_skill"],
                "daily_seasonal_MSE_skill": metrics["skill"]["daily_seasonal"]["MSE_skill"],
            }
        ],
        [
            "split",
            "windows",
            "n_samples",
            "MAE",
            "MSE",
            "RMSE",
            "CRPS",
            "CRPS_sum",
            "last_value_MSE_skill",
            "daily_seasonal_MSE_skill",
        ],
    )
    baseline_rows = []
    for name, baseline in baseline_metrics.items():
        baseline_rows.append(
            {
                "baseline": name,
                "MAE": baseline["MAE"],
                "MSE": baseline["MSE"],
                "RMSE": baseline["RMSE"],
                "CRPS": baseline["CRPS"],
                "CRPS_sum": baseline["CRPS_sum"],
                "MSE_skill": metrics["skill"][name]["MSE_skill"],
                "RMSE_delta": metrics["skill"][name]["RMSE_delta"],
            }
        )
    _write_csv(
        metrics_dir / "baseline_metrics.csv",
        baseline_rows,
        ["baseline", "MAE", "MSE", "RMSE", "CRPS", "CRPS_sum", "MSE_skill", "RMSE_delta"],
    )
    print(
        f"[Eval:{split_name}] windows={metrics['windows']} "
        f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
        f"CRPS={metrics['CRPS']:.4f} CRPS_sum={metrics['CRPS_sum']:.4f}"
    )
    return metrics


def _phase0_candidate_methods(config: dict) -> list[str]:
    methods = config.get("phase0_candidate_methods", None)
    if methods is None:
        methods = AVAILABLE_STRICT_METHODS
    if not isinstance(methods, list) or not methods:
        raise ValueError("phase0_candidate_methods must be a non-empty list when provided")
    out = []
    for method in methods:
        method = str(method)
        if method not in AVAILABLE_STRICT_METHODS:
            raise ValueError(
                f"Phase 0 candidate {method!r} is not an available strict method; "
                f"allowed={AVAILABLE_STRICT_METHODS}"
            )
        out.append(method)
    return out


def _build_protocol(method: str, target_dim: int, context_len: int, pred_len: int, device: torch.device, config: dict):
    cls = FULL_PROTOCOL_REGISTRY[method]
    return cls(
        target_dim=int(target_dim),
        context_len=int(context_len),
        pred_len=int(pred_len),
        device=device,
        **_protocol_overrides(config, method),
    ).to(device)


@torch.no_grad()
def _sample_protocol_all_channels(protocol, batch: FullProtocolBatch, n_samples: int) -> torch.Tensor:
    sample_count = int(n_samples)
    method_name = str(getattr(protocol, "method_name", ""))

    if hasattr(protocol, "_sample_full"):
        samples = protocol._sample_full(batch, n_samples=sample_count)
        return samples[:, :, :, protocol.context_len :]

    if method_name in {"CSDI", "PriSTI"} and hasattr(protocol, "model") and hasattr(protocol, "_make_batch"):
        if method_name == "CSDI" and hasattr(protocol.model, "target_dim_base"):
            protocol.model.target_dim = protocol.model.target_dim_base
        samples, _, _, _, _ = protocol.model.evaluate(protocol._make_batch(batch), n_samples=sample_count)
        return samples[:, :, :, protocol.context_len :]

    if method_name == "DiffSTG":
        _, condition = protocol._prepare(batch)
        samples = protocol.model.evaluate(condition, n_samples=sample_count)
        return samples[:, :, 0, :, protocol.context_len :]

    if method_name == "Diffusion-TS":
        series = protocol._series(batch)
        batch_size = series.size(0)
        target = series.repeat_interleave(sample_count, dim=0)
        target[:, protocol.context_len :, :] = 0.0
        known_mask = torch.zeros_like(target, dtype=torch.bool)
        known_mask[:, : protocol.context_len, :] = True
        if protocol.model.fast_sampling:
            samples = protocol.model.fast_sample_infill(
                target.shape,
                target=target,
                sampling_timesteps=protocol.model.sampling_timesteps,
                partial_mask=known_mask,
                model_kwargs={"coef": 0.0, "learning_rate": 0.0},
            )
        else:
            samples = protocol.model.sample_infill(
                target.shape,
                target=target,
                partial_mask=known_mask,
                model_kwargs={"coef": 0.0, "learning_rate": 0.0},
            )
        samples = samples.reshape(batch_size, sample_count, protocol.context_len + protocol.pred_len, protocol.target_dim)
        return samples[:, :, protocol.context_len :, :].permute(0, 1, 3, 2).contiguous()

    if method_name == "TimeGrad":
        args = protocol._prepare(batch)
        old_n = protocol.pred_net.num_parallel_samples
        protocol.pred_net.num_parallel_samples = sample_count
        try:
            samples = protocol.pred_net(
                target_dimension_indicator=args[0],
                past_time_feat=args[1],
                past_target_cdf=args[2],
                past_observed_values=args[3],
                past_is_pad=args[4],
                future_time_feat=args[5],
            )
        finally:
            protocol.pred_net.num_parallel_samples = old_n
        return samples.permute(0, 1, 3, 2).contiguous()

    channel_samples = [
        protocol.sample_channel(batch, channel_idx=idx, n_samples=sample_count).unsqueeze(2)
        for idx in range(protocol.target_dim)
    ]
    return torch.cat(channel_samples, dim=2)


def _evaluate_protocol_channel_rmses(
    protocol,
    dataset,
    *,
    target_names: list[str],
    device: torch.device,
    batch_size: int,
    n_samples: int,
    max_windows: int | None,
) -> list[float]:
    """Evaluate one multichannel expert once, then score each channel independently."""
    eval_dataset = _limit_dataset(dataset, max_windows)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    acc = _new_metric_accumulator(len(target_names))
    protocol.eval()
    with torch.no_grad():
        for batch in loader:
            protocol_batch = _collate_batch(batch, device)
            samples = _sample_protocol_all_channels(protocol, protocol_batch, n_samples=int(n_samples))
            pred_bcl = samples.mean(dim=1)
            true_bcl = protocol_batch.future
            pred_raw = _inverse_batch(eval_dataset, pred_bcl)
            true_raw = _inverse_batch(eval_dataset, true_bcl)
            _update_metric_accumulator(acc, pred_raw, true_raw)

    metrics = _finalize_metric_accumulator(acc, target_names)
    return [float(row["RMSE"]) for row in metrics["channel_metrics"]]


def run_phase0_channel_selection(
    *,
    config: dict,
    train_dataset,
    val_dataset,
    target_names: list[str],
    context_len: int,
    pred_len: int,
    device: torch.device,
    save_dir: Path,
) -> dict[int, str]:
    methods = _phase0_candidate_methods(config)
    epochs = int(config.get("phase0_epochs", 1))
    lr = float(config.get("phase0_lr", config.get("lr", 1e-3)))
    batch_size = int(config.get("phase0_batch_size", config.get("batch_size", 1)))
    eval_batch_size = int(config.get("phase0_eval_batch_size", config.get("eval_batch_size", batch_size)))
    eval_samples = int(config.get("phase0_eval_samples", config.get("eval_samples", 1)))
    train_max_windows = config.get("phase0_train_max_windows", None)
    eval_max_windows = config.get("phase0_eval_max_windows", config.get("val_eval_max_windows", None))
    train_max_windows = None if train_max_windows is None else int(train_max_windows)
    eval_max_windows = None if eval_max_windows is None else int(eval_max_windows)
    grad_clip = config.get("grad_clip_norm", None)

    phase0_train_dataset = _limit_dataset(train_dataset, train_max_windows)
    loader = DataLoader(phase0_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    rows = []
    best_by_channel = {
        idx: {"method": None, "rmse": float("inf")}
        for idx in range(len(target_names))
    }

    old_candidate_runs = len(target_names) * len(methods)
    new_candidate_runs = len(methods)
    print(
        f"[Phase0] channel expert selection | mode={PHASE0_SELECTION_MODE} | "
        f"methods={methods} | channels={len(target_names)} | epochs={epochs} | "
        f"train_windows={len(phase0_train_dataset)}"
    )
    print(
        f"[Phase0] acceleration: old channel-wise candidate runs={old_candidate_runs}, "
        f"current multichannel candidate runs={new_candidate_runs}. "
        "Final assignment is still selected independently per channel."
    )
    for method in methods:
        # Acceleration only: train one full multichannel expert and reuse it to
        # score all channels. This preserves channel-wise expert selection.
        print(f"[Phase0] method={method} training once for {len(target_names)} channels...")
        protocol = _build_protocol(method, len(target_names), context_len, pred_len, device, config)
        optimizer = torch.optim.Adam(protocol.parameters(), lr=lr)
        train_loss = float("nan")
        for epoch in range(1, epochs + 1):
            protocol.train()
            running = 0.0
            count = 0
            for batch in loader:
                protocol_batch = _collate_batch(batch, device)
                loss = protocol.training_loss(protocol_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(protocol.parameters(), float(grad_clip))
                optimizer.step()
                running += float(loss.detach().cpu()) * protocol_batch.past.size(0)
                count += int(protocol_batch.past.size(0))
            train_loss = running / max(1, count)
            print(f"[Phase0] method={method} epoch={epoch}/{epochs} loss={train_loss:.6f}")

        rmses = _evaluate_protocol_channel_rmses(
            protocol,
            val_dataset,
            target_names=target_names,
            device=device,
            batch_size=eval_batch_size,
            n_samples=eval_samples,
            max_windows=eval_max_windows,
        )
        for channel_idx, (channel_name, rmse) in enumerate(zip(target_names, rmses)):
            rows.append(
                {
                    "channel_idx": int(channel_idx),
                    "channel": str(channel_name),
                    "method": str(method),
                    "phase0_mode": PHASE0_SELECTION_MODE,
                    "phase0_train_loss": float(train_loss),
                    "phase0_val_RMSE": float(rmse),
                }
            )
            print(f"[Phase0] channel={channel_name} method={method} val_RMSE={rmse:.4f}")
            if rmse < best_by_channel[channel_idx]["rmse"]:
                best_by_channel[channel_idx] = {"method": method, "rmse": float(rmse)}

        del protocol
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assignment: dict[int, str] = {}
    for channel_idx, channel_name in enumerate(target_names):
        best = best_by_channel[channel_idx]
        if best["method"] is None:
            raise RuntimeError(f"Phase 0 failed to select a method for channel {channel_name}")
        assignment[channel_idx] = str(best["method"])
        print(
            f"[Phase0] SELECT channel={channel_name} method={best['method']} "
            f"val_RMSE={float(best['rmse']):.4f}"
        )

    _write_csv(
        save_dir / "phase0_channel_selection.csv",
        rows,
        ["channel_idx", "channel", "method", "phase0_mode", "phase0_train_loss", "phase0_val_RMSE"],
    )
    payload = {
        "phase0_mode": PHASE0_SELECTION_MODE,
        "phase0_mode_description": (
            "Each candidate expert is trained once on the full multichannel target set; "
            "validation RMSE is computed per channel; the lowest-RMSE expert is assigned "
            "independently to each channel. This accelerates Phase 0 but does not change "
            "the final hard channel routing objective."
        ),
        "candidate_training_runs": {
            "old_channel_wise": old_candidate_runs,
            "current_multichannel": new_candidate_runs,
        },
        "selection_metric": "val_RMSE",
        "candidate_methods": methods,
        "target_names": list(target_names),
        "assignment": {
            str(idx): {
                "channel": target_names[idx],
                "method": method,
                "best_val_RMSE": min(
                    float(row["phase0_val_RMSE"])
                    for row in rows
                    if int(row["channel_idx"]) == idx and str(row["method"]) == method
                ),
            }
            for idx, method in assignment.items()
        },
        "rows": rows,
    }
    with open(save_dir / "phase0_channel_selection.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict full-protocol channel MoE training entrypoint.")
    parser.add_argument("--config", default="configs/full_protocol_etth1_server.json")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = _load_json(config_path)
    _validate_config(config, config_path)
    _set_seed(int(config.get("seed", 42)))

    device = _resolve_device(config)
    csv_path = (ROOT / str(config.get("csv_path", "csv_data/ETTh1.csv"))).resolve()
    context_len = int(config.get("context_len", 24))
    pred_len = int(config.get("pred_len", 24))
    batch_size = int(config.get("batch_size", 4))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    eval_every = int(config.get("eval_every", 5))
    eval_samples = int(config.get("eval_samples", 1))
    baseline_period = int(config.get("baseline_period", 24))
    val_eval_max_windows = config.get("val_eval_max_windows", 32)
    test_eval_max_windows = config.get("test_eval_max_windows", 64)
    val_eval_max_windows = None if val_eval_max_windows is None else int(val_eval_max_windows)
    test_eval_max_windows = None if test_eval_max_windows is None else int(test_eval_max_windows)

    dataset = _build_dataset(config, csv_path, "train")
    val_dataset = _build_dataset(config, csv_path, "val", train_dataset=dataset)
    test_dataset = _build_dataset(config, csv_path, "test", train_dataset=dataset)

    target_names = list(getattr(dataset.dataset if isinstance(dataset, Subset) else dataset, "target_cols"))
    run_name = str(config.get("run_tag", "full_protocol_moe"))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = ROOT / "results" / f"{run_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "strict_status.json", "w", encoding="utf-8") as f:
        json.dump(STRICT_REPRODUCTION_STATUS, f, indent=2, ensure_ascii=False)

    if bool(config.get("phase0_channel_selection", True)):
        assignment = run_phase0_channel_selection(
            config=config,
            train_dataset=dataset,
            val_dataset=val_dataset,
            target_names=target_names,
            context_len=context_len,
            pred_len=pred_len,
            device=device,
            save_dir=save_dir,
        )
    else:
        assignment = _normalize_assignment(config.get("channel_methods", None), target_names)

    protocol_kwargs = _protocol_kwargs_for_assignment(config, assignment)

    with open(save_dir / "channel_assignment.json", "w", encoding="utf-8") as f:
        payload = {
            "target_names": target_names,
            "assignment": {str(idx): {"channel": target_names[idx], "method": method} for idx, method in assignment.items()},
        }
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _write_experiment_metadata(
        save_dir,
        config=config,
        config_path=config_path,
        device=device,
        csv_path=csv_path,
        train_dataset=dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        target_names=target_names,
        assignment=assignment,
        protocol_kwargs=protocol_kwargs,
    )

    model = FullProtocolMoEDiffuser(
        target_dim=len(target_names),
        context_len=context_len,
        pred_len=pred_len,
        device=device,
        channel_to_method=assignment,
        protocol_kwargs=protocol_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history_rows = []
    epochs = int(config.get("epochs", 1))
    best_val_rmse = float("inf")
    best_epoch = None
    best_path = save_dir / "model_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in loader:
            protocol_batch = _collate_batch(batch, device)
            loss = model.training_loss(protocol_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = config.get("grad_clip_norm", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            running += float(loss.detach().cpu()) * protocol_batch.past.size(0)
            count += int(protocol_batch.past.size(0))
        avg_loss = running / max(1, count)
        row = {"epoch": epoch, "loss": avg_loss, "val_RMSE": ""}
        print(f"[FullProtocol] Epoch {epoch}/{epochs} | loss={avg_loss:.6f}")

        should_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if should_eval:
            val_metrics = _evaluate_model(
                model,
                val_dataset,
                split_name=f"val_epoch_{epoch}",
                device=device,
                batch_size=eval_batch_size,
                n_samples=eval_samples,
                max_windows=val_eval_max_windows,
                save_dir=save_dir,
                target_names=target_names,
                baseline_period=baseline_period,
            )
            if val_metrics is not None:
                val_rmse = float(val_metrics["RMSE"])
                row["val_RMSE"] = val_rmse
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_path)
                    with open(save_dir / "best_val_metric.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "selection_metric": "RMSE",
                                "selection_split": "val",
                                "epoch": int(best_epoch),
                                "RMSE": float(best_val_rmse),
                                "eval_windows": val_metrics["windows"],
                                "n_samples": int(eval_samples),
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    print(f"[Best] epoch={best_epoch} val_RMSE={best_val_rmse:.4f}")
        history_rows.append(row)

    with open(save_dir / "loss_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_RMSE"])
        writer.writeheader()
        writer.writerows(history_rows)
    torch.save(model.state_dict(), save_dir / "model_last.pt")
    if best_path.exists():
        _load_state_dict_compat(model, best_path, device)
    else:
        best_epoch = epochs
        best_val_rmse = float("nan")
        torch.save(model.state_dict(), best_path)

    final_val = _evaluate_model(
        model,
        val_dataset,
        split_name="FinalVal",
        device=device,
        batch_size=eval_batch_size,
        n_samples=eval_samples,
        max_windows=val_eval_max_windows,
        save_dir=save_dir,
        target_names=target_names,
        baseline_period=baseline_period,
    )
    final_test = _evaluate_model(
        model,
        test_dataset,
        split_name="FinalTest",
        device=device,
        batch_size=eval_batch_size,
        n_samples=eval_samples,
        max_windows=test_eval_max_windows,
        save_dir=save_dir,
        target_names=target_names,
        baseline_period=baseline_period,
    )
    with open(save_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_RMSE": best_val_rmse,
                "FinalVal": final_val,
                "FinalTest": final_test,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Done] {save_dir}")


if __name__ == "__main__":
    main()
