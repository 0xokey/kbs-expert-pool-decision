import contextlib
import csv
import datetime
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from network.moe import MoEDiffuser


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _bcl_to_lc(x_bcl):
    x_bcl = _to_numpy(x_bcl)
    if x_bcl.ndim != 3:
        raise RuntimeError(f"_bcl_to_lc expects [B,C,L], got {x_bcl.shape}")
    return x_bcl[0].transpose(1, 0)


def _bcl_to_blc(x_bcl):
    x_bcl = _to_numpy(x_bcl)
    if x_bcl.ndim != 3:
        raise RuntimeError(f"_bcl_to_blc expects [B,C,L], got {x_bcl.shape}")
    return np.transpose(x_bcl, (0, 2, 1))


def _safe_tag(value):
    tag = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value).strip())
    return tag.strip("_") or "eval"


def _jsonable_metrics(metrics):
    return {k: float(v) for k, v in metrics.items()}


def _stringify_time_value(value):
    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.datetime64):
            return np.datetime_as_string(arr, unit="s")
    except Exception:
        pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


def _get_target_names(dataset, target_idx):
    target_cols = getattr(dataset, "target_cols", None)
    if isinstance(target_cols, (list, tuple)) and len(target_cols) == len(target_idx):
        return [str(v) for v in target_cols]

    feature_cols = getattr(dataset, "feature_cols", None)
    if isinstance(feature_cols, (list, tuple)):
        names = []
        for idx in target_idx:
            names.append(str(feature_cols[idx]) if 0 <= idx < len(feature_cols) else f"ch{idx}")
        return names

    return [f"ch{idx}" for idx in target_idx]


def _write_csv_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_horizon_boundaries(config: dict):
    raw = config.get("horizon_boundaries", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        values = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                values.append(int(part))
        return values if values else None
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw if int(v) > 0]
        return values if values else None
    return None


def _resolve_target_loss_weights(config: dict, target_names) -> torch.Tensor | None:
    raw = config.get("target_loss_weights", None)
    if raw is None:
        return None

    names = [str(name) for name in (target_names or [])]
    if len(names) == 0:
        return None

    weights = None
    if isinstance(raw, dict):
        weights = [float(raw.get(name, 1.0)) for name in names]
    elif isinstance(raw, (list, tuple)):
        weights = [float(v) for v in raw]
        if len(weights) != len(names):
            raise ValueError(
                f"target_loss_weights length mismatch: got {len(weights)}, expected {len(names)}"
            )
    else:
        raise ValueError("target_loss_weights must be a dict keyed by target name or a list")

    arr = np.asarray(weights, dtype=np.float32)
    if np.any(arr <= 0):
        raise ValueError(f"target_loss_weights must be positive, got {weights}")
    if np.allclose(arr, 1.0):
        return None
    return torch.tensor(arr, dtype=torch.float32)


def _compute_target_recon_loss(pred_noise, noise, target_idx, target_loss_weights: torch.Tensor | None = None):
    pred_target = pred_noise[:, target_idx, :]
    noise_target = noise[:, target_idx, :]
    if target_loss_weights is None:
        return F.mse_loss(pred_target, noise_target)

    weights = target_loss_weights.to(device=pred_target.device, dtype=pred_target.dtype)
    if int(weights.numel()) != int(pred_target.shape[1]):
        raise ValueError(
            f"target_loss_weights size mismatch: got {int(weights.numel())}, "
            f"expected {int(pred_target.shape[1])}"
        )

    channel_mse = (pred_target - noise_target).pow(2).mean(dim=(0, 2))
    return (channel_mse * weights).sum() / weights.sum().clamp_min(1e-8)


def _distribution_entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-(arr * np.log(arr + 1e-12)).sum())


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * (current / total)))
    filled = max(0, min(filled, width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class _ProgressReporter:
    def __init__(self, label: str, total: int, min_interval_sec: float = 30.0):
        self.label = str(label)
        self.total = max(1, int(total))
        self.min_interval_sec = float(max(1.0, min_interval_sec))
        self.start_time = time.time()
        self.last_report_time = 0.0
        self.last_completed = -1

    def update(self, completed: int, extra: str = "", force: bool = False):
        completed = max(0, min(int(completed), self.total))
        now = time.time()

        should_report = bool(force) or completed <= 1 or completed >= self.total
        if not should_report and completed == self.last_completed:
            return
        if not should_report and (now - self.last_report_time) < self.min_interval_sec:
            return

        elapsed = now - self.start_time
        rate = completed / max(elapsed, 1e-8)
        eta = float("inf") if completed <= 0 else (self.total - completed) / max(rate, 1e-8)
        percent = 100.0 * completed / self.total
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        extra_text = f" | {extra}" if extra else ""
        print(
            f"   [{stamp}] {self.label} {_progress_bar(completed, self.total)} "
            f"{completed}/{self.total} ({percent:.1f}%) | "
            f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}{extra_text}"
        )
        self.last_report_time = now
        self.last_completed = completed


def select_eval_indices(total_len: int, max_windows: int | None = None, seed: int = 0):
    total_len = int(total_len)
    if total_len <= 0:
        return []

    if max_windows is None or int(max_windows) >= total_len:
        return list(range(total_len))

    rng = np.random.default_rng(int(seed))
    indices = rng.choice(total_len, size=int(max_windows), replace=False)
    return sorted(int(i) for i in indices.tolist())


def _limit_eval_indices(indices, max_windows: int | None, seed: int) -> list[int]:
    clean = [int(i) for i in (indices or [])]
    if max_windows is None:
        return clean
    max_windows = int(max_windows)
    if max_windows <= 0 or len(clean) <= max_windows:
        return clean
    selected_pos = select_eval_indices(len(clean), max_windows, seed=seed)
    return [clean[pos] for pos in selected_pos]


@contextlib.contextmanager
def _temporary_expert_gate_scale(model, expert_idx: int, new_value: float):
    if not isinstance(model, MoEDiffuser):
        yield
        return

    expert_idx = int(expert_idx)
    if expert_idx < 0 or expert_idx >= int(model.num_experts):
        yield
        return

    original = model.expert_gate_scale[expert_idx].detach().clone()
    model.expert_gate_scale[expert_idx] = torch.as_tensor(
        float(new_value),
        dtype=model.expert_gate_scale.dtype,
        device=model.expert_gate_scale.device,
    )
    try:
        yield
    finally:
        model.expert_gate_scale[expert_idx] = original


def set_global_seed(seed: int, deterministic: bool = True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def make_cpu_generator(seed: int):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def is_amp_enabled(config: dict, device: torch.device) -> bool:
    return bool(config.get("use_amp", True)) and device.type == "cuda"


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_grad_clip_norm(config: dict) -> float | None:
    value = config.get("grad_clip_norm", 1.0)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _optimizer_params(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param is not None and param.requires_grad:
                yield param


def optimizer_backward_step(loss, optimizer, scaler, grad_clip_norm: float | None = None):
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            params = [p for p in _optimizer_params(optimizer) if p.grad is not None]
            if len(params) > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    if grad_clip_norm is not None:
        params = [p for p in _optimizer_params(optimizer) if p.grad is not None]
        if len(params) > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
    optimizer.step()
