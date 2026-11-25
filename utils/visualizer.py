# utils/visualizer.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(x):
    """torch/tensor -> numpy"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _ensure_lc(x, name="x"):
    """
    统一成 [L, C] 的 numpy
    支持输入:
      - [L]
      - [L, C]
      - [C, L]
      - [B, C, L] / [B, L, C] (会取 B=0)
    """
    x = _to_numpy(x)

    if x.ndim == 1:
        return x.reshape(-1, 1)

    if x.ndim == 3:
        x = x[0]  # pick batch=0
        if x.ndim != 2:
            raise ValueError(f"{name}: expect after batch pick 2D, got {x.shape}")

    if x.ndim != 2:
        raise ValueError(f"{name}: expect 1D/2D/3D, got {x.shape}")

    # heuristic: [C,L] -> [L,C]
    if x.shape[0] <= 32 and x.shape[1] > x.shape[0]:
        x = x.T

    return x  # [L,C]


def _sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:80] if len(s) > 80 else s


def _is_datetime_like(arr) -> bool:
    try:
        a = np.asarray(arr)
        return np.issubdtype(a.dtype, np.datetime64)
    except Exception:
        return False


def _default_channel_pool(dataset, C: int, use_target_only: bool):
    """
    默认通道池：
    - 如果 dataset 有 target_indices 且 use_target_only=True -> 用 target_indices
    - 否则 -> range(C)
    """
    if use_target_only and dataset is not None and hasattr(dataset, "target_indices"):
        idxs = list(getattr(dataset, "target_indices") or [])
        idxs = [i for i in idxs if 0 <= i < C]
        if len(idxs) > 0:
            return idxs
    return list(range(C))


def _get_channel_name(dataset, c: int, prefer_target: bool = True):
    """
    给通道一个可读名字：
    - 优先：target_cols 对应的 feature_cols index
    - 其次：feature_cols[c]
    - 否则：ch{c}
    """
    if dataset is None:
        return f"ch{c}"

    feat_cols = getattr(dataset, "feature_cols", None)
    if isinstance(feat_cols, (list, tuple)) and 0 <= c < len(feat_cols):
        name = feat_cols[c]
    else:
        name = f"ch{c}"

    if prefer_target and hasattr(dataset, "target_cols") and hasattr(dataset, "target_indices"):
        t_cols = getattr(dataset, "target_cols", None)
        t_idxs = getattr(dataset, "target_indices", None)
        if isinstance(t_cols, (list, tuple)) and isinstance(t_idxs, (list, tuple)):
            for ti, tc in zip(t_idxs, t_cols):
                if ti == c:
                    return str(tc)

    return str(name)


def save_prediction_plot(
    history,
    true_future,
    pred_future,
    epoch,
    save_dir,
    channel_indices=None,
    max_channels=None,      # ✅ None=自动（目标变量全画）
    dataset=None,
    sample_idx=None,
    use_target_only=True,
    dpi=150,
):
    """
    兼容单变量/多变量的画图函数。
    - history / true_future / pred_future 最终都会被规整成 [L,C]
    - 若提供 dataset + sample_idx 且 dataset.get_time_window 存在：横轴用真实时间
    - 若 dataset 有 target_indices 且 use_target_only=True：默认只画 target 通道池
    """
    os.makedirs(save_dir, exist_ok=True)

    hist = _ensure_lc(history, "history")          # [Lh,C]
    true = _ensure_lc(true_future, "true_future")  # [Lf,C]
    pred = _ensure_lc(pred_future, "pred_future")  # [Lf,C]

    Lh, Ch = hist.shape
    Lt, Ct = true.shape
    Lp, Cp = pred.shape

    if Ct != Cp:
        raise RuntimeError(f"true/pred channel mismatch: true={true.shape}, pred={pred.shape}")

    C = Ct
    if Ch != C:
        C = min(Ch, Ct, Cp)

    # === 1) 选择要画的通道 ===
    pool = _default_channel_pool(dataset, C, use_target_only=use_target_only)

    # max_channels=None -> 自动画完 pool（目标变量全画）
    if max_channels is None:
        max_channels = len(pool) if len(pool) > 0 else C
    else:
        max_channels = int(max_channels)

    if channel_indices is None:
        if len(pool) == 0:
            pool = list(range(C))
        channel_indices = pool[:max_channels]
    else:
        channel_indices = [int(c) for c in channel_indices if 0 <= int(c) < C]
        if len(channel_indices) == 0:
            channel_indices = pool[:max_channels] if len(pool) else [0]
        else:
            channel_indices = channel_indices[:max_channels]

    # === 2) 时间轴：优先用真实时间 ===
    time_past = np.arange(0, Lh)
    time_future = np.arange(Lh, Lh + Lt)
    if dataset is not None and sample_idx is not None and hasattr(dataset, "get_time_window"):
        try:
            t_past, t_fut = dataset.get_time_window(sample_idx)
            if len(t_past) == Lh and len(t_fut) == Lt:
                time_past = np.asarray(t_past)
                time_future = np.asarray(t_fut)
        except Exception:
            pass

    # === 3) 作图：每个通道一张（清晰）===
    saved_paths = []
    for c in channel_indices:
        var_name = _get_channel_name(dataset, c, prefer_target=True)
        tag = f"{epoch}"
        safe_var = _sanitize_filename(var_name)

        fig = plt.figure(figsize=(11, 4))
        ax = fig.add_subplot(111)

        ax.plot(time_past, hist[:Lh, c], label="History")
        ax.plot(time_future, true[:Lt, c], label="Ground Truth", linestyle="--")
        ax.plot(time_future, pred[:Lt, c], label="Prediction")

        ax.set_title(f"Forecast | {tag} | {var_name}")
        ax.set_xlabel("time" if _is_datetime_like(time_past) else "time index")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # datetime 横轴更友好
        if _is_datetime_like(time_past):
            fig.autofmt_xdate(rotation=30)

        fname = f"pred_{_sanitize_filename(tag)}_{safe_var}.png"
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        saved_paths.append(save_path)

    print(f"   Plots saved to: {save_dir} (tag={epoch}, channels={channel_indices})")
    return saved_paths


def save_series_overview_plot(
    time_axis,
    true_series,
    pred_series,
    title,
    save_path,
    dpi=150,
):
    """Save a single continuous target-series overview plot."""
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    t = np.asarray(time_axis)
    y_true = np.asarray(true_series).reshape(-1)
    y_pred = np.asarray(pred_series).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"true/pred length mismatch: true={y_true.shape}, pred={y_pred.shape}")
    if len(t) != len(y_true):
        t = np.arange(len(y_true))

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    ax.plot(t, y_true, label="Ground Truth")
    ax.plot(t, y_pred, label="Prediction")

    ax.set_title(title)
    ax.set_xlabel("time" if _is_datetime_like(t) else "time index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if _is_datetime_like(t):
        fig.autofmt_xdate(rotation=30)

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path
