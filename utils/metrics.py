import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def resolve_horizon_ranges(pred_len: int, boundaries: list[int] | None = None) -> list[tuple[str, int, int]]:
    pred_len = int(max(1, pred_len))
    if boundaries is None or len(boundaries) == 0:
        if pred_len <= 24:
            boundaries = [pred_len]
        elif pred_len <= 72:
            boundaries = [24, pred_len]
        else:
            boundaries = [24, 72, pred_len]

    clean = sorted({int(v) for v in boundaries if int(v) > 0})
    if len(clean) == 0 or clean[-1] < pred_len:
        clean.append(pred_len)

    ranges: list[tuple[str, int, int]] = []
    start = 0
    for end in clean:
        end = int(min(pred_len, end))
        if end <= start:
            continue
        label = f"{start + 1}-{end}"
        ranges.append((label, start, end))
        start = end
        if start >= pred_len:
            break
    return ranges


class MetricsCalculator:
    @staticmethod
    def crps(ensemble: np.ndarray, y_true: np.ndarray) -> float:
        """Continuous Ranked Probability Score for ensemble predictions.

        ensemble: [N, ...] — N independent draws from the predictive distribution
        y_true:   [...]     — ground truth

        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]   (X, X' i.i.d. ~ predictive)
        """
        ensemble = np.asarray(ensemble, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        N = ensemble.shape[0]
        if N < 2:
            return float(np.mean(np.abs(ensemble - y_true)))
        # Term 1: mean absolute error per ensemble member
        term1 = np.mean(np.abs(ensemble - y_true))
        # Term 2: 0.5 * mean pairwise absolute difference over samples and all target dimensions
        diffs = 0.0
        elem_count = float(max(1, np.prod(ensemble.shape[1:])))
        for i in range(N):
            diffs += np.sum(np.abs(ensemble[i] - ensemble))
        term2 = 0.5 * diffs / (N * N * elem_count)
        return float(term1 - term2)

    @staticmethod
    def compute(y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        epsilon = 1e-8
        ape = np.abs((y_true - y_pred) / (y_true + epsilon))
        ape = np.clip(ape, 0, 10.0)
        mape = np.mean(ape) * 100
        mspe = np.mean(np.square(np.abs((y_true - y_pred) / (y_true + epsilon)))) * 100
        smape = np.mean(
            2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
        ) * 100

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "MSPE": mspe,
            "sMAPE": smape,
            "R2": r2,
        }

    @staticmethod
    def compute_horizon_metrics(
        y_true,
        y_pred,
        horizon_ranges: list[tuple[str, int, int]] | None = None,
    ) -> list[dict]:
        true_arr = np.asarray(y_true)
        pred_arr = np.asarray(y_pred)

        if true_arr.ndim == 1:
            true_arr = true_arr.reshape(-1, 1)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)

        pred_len = int(true_arr.shape[0])
        if pred_arr.shape[0] != pred_len:
            raise RuntimeError(f"horizon metric length mismatch: true={true_arr.shape}, pred={pred_arr.shape}")

        if horizon_ranges is None:
            horizon_ranges = resolve_horizon_ranges(pred_len)

        rows = []
        for label, start, end in horizon_ranges:
            start = int(max(0, start))
            end = int(min(pred_len, end))
            if end <= start:
                continue
            metrics = MetricsCalculator.compute(true_arr[start:end, :], pred_arr[start:end, :])
            rows.append({
                "horizon": str(label),
                "start_step": int(start + 1),
                "end_step": int(end),
                "MAE": float(metrics["MAE"]),
                "MSE": float(metrics["MSE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAPE": float(metrics["MAPE"]),
                "MSPE": float(metrics["MSPE"]),
                "sMAPE": float(metrics["sMAPE"]),
                "R2": float(metrics["R2"]),
            })
        return rows


def compute_overall_and_per_channel(y_true, y_pred, target_idx=None):
    true_lc = np.asarray(y_true)
    pred_lc = np.asarray(y_pred)

    if true_lc.ndim == 1:
        true_lc = true_lc.reshape(-1, 1)
    if pred_lc.ndim == 1:
        pred_lc = pred_lc.reshape(-1, 1)

    if target_idx is not None and len(target_idx) > 0:
        true_eval = true_lc[:, target_idx]
        pred_eval = pred_lc[:, target_idx]
    else:
        true_eval = true_lc
        pred_eval = pred_lc

    overall = MetricsCalculator.compute(true_eval.reshape(-1), pred_eval.reshape(-1))

    per_mae, per_mse, per_rmse = [], [], []
    for c in range(true_eval.shape[1]):
        channel_metrics = MetricsCalculator.compute(true_eval[:, c], pred_eval[:, c])
        per_mae.append(float(channel_metrics["MAE"]))
        per_mse.append(float(channel_metrics["MSE"]))
        per_rmse.append(float(channel_metrics["RMSE"]))

    return (
        float(overall["MAE"]),
        float(overall["MSE"]),
        float(overall["RMSE"]),
        per_mae,
        per_mse,
        per_rmse,
    )
