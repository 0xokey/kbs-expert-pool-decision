import numpy as np


def _ensure_lc(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise RuntimeError(f"expected [L,C], got {arr.shape}")
    return arr


def apply_linear_drift_correction(
    history_lc,
    pred_lc,
    target_indices=None,
    tail_window: int = 24,
    alpha: float = 0.75,
):
    """
    Align the prediction head to the recent history level, then decay the
    correction linearly over the prediction horizon.
    """
    hist = _ensure_lc(history_lc)
    pred = _ensure_lc(pred_lc).copy()

    if hist.shape[1] != pred.shape[1]:
        raise RuntimeError(f"history/pred channel mismatch: {hist.shape} vs {pred.shape}")

    if hist.shape[0] == 0 or pred.shape[0] == 0:
        return pred

    if target_indices is None or len(target_indices) == 0:
        target_indices = list(range(pred.shape[1]))
    else:
        target_indices = [int(i) for i in target_indices if 0 <= int(i) < pred.shape[1]]

    if len(target_indices) == 0:
        return pred

    k = max(1, min(int(tail_window), hist.shape[0], pred.shape[0]))
    decay = np.linspace(1.0, 0.0, num=pred.shape[0], dtype=np.float32).reshape(-1, 1)

    hist_level = hist[-k:, target_indices].mean(axis=0, keepdims=True)
    pred_level = pred[:k, target_indices].mean(axis=0, keepdims=True)
    offset = hist_level - pred_level

    pred[:, target_indices] = pred[:, target_indices] + float(alpha) * decay * offset
    return pred
