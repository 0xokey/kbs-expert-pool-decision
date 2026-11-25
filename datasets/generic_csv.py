import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class GenericCSVDataset(Dataset):
    """
    Time-series dataset for forecasting (multivariate):

    - Auto-detect / parse time column (date/time/timestamp)
    - Optional time feature engineering (+ cyclical sin/cos)
    - feature_cols: model input channels (base numeric + time engineered)
    - target_cols: what you REALLY want to predict/plot/metric (default: base numeric only)
    - train/val/test can reuse scaler/feature_cols/target_cols from train to keep consistency
    - Return shape: [C,L] if channels_first=True else [L,C]
    - Optional sample-level coarse labels loaded from label_path
    """

    _TIME_ENGINEERED_SET = {
        "hour", "minute", "dayofweek", "day", "month", "dayofyear",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    }

    def __init__(
        self,
        seq_len: int,
        context_len: int | None = None,
        pred_len: int | None = None,
        csv_file_path: str | None = None,
        mode: str = "train",
        channels_first: bool = True,
        time_col: str | None = None,
        add_time_features: bool = True,
        add_cyc_features: bool = True,
        feature_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
        scaler: MinMaxScaler | None = None,
        sort_by_time: bool = True,
        label_path: str | None = None,
        target_only_diffusion: bool = False,
        split_borders: list[int] | None = None,
        split_ratios: list[float] | None = None,
        scaler_type: str = "minmax",
    ):
        if context_len is None:
            context_len = seq_len
        if pred_len is None:
            pred_len = seq_len

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.pred_len
        self.mode = mode
        self.channels_first = channels_first
        self.label_path = label_path
        self.target_only_diffusion = bool(target_only_diffusion)
        self.split_borders = [int(v) for v in split_borders] if split_borders else None
        self.split_ratios = [float(v) for v in split_ratios] if split_ratios else [0.8, 0.1, 0.1]
        self.scaler_type = str(scaler_type or "minmax").strip().lower()
        if self.scaler_type not in {"minmax", "standard"}:
            raise ValueError(f"scaler_type must be 'minmax' or 'standard', got {scaler_type!r}")

        self.time_col: str | None = None
        self.time_values_all = None
        self.time_slice = None

        self.feature_cols: list[str] = []
        self.target_cols: list[str] = []
        self.target_indices: list[int] = []
        self.target_feature_indices: list[int] = []
        self.known_feature_cols: list[str] = []
        self.known_feature_indices: list[int] = []
        self.model_feature_cols: list[str] = []
        self.known_future_channels: int = 0
        self.full_num_channels: int = 0

        self.sample_labels = None  # np.ndarray or None

        # =========================
        # A) 无文件 -> 随机数据
        # =========================
        if csv_file_path is None:
            raw = np.random.rand(2000, 7).astype(np.float32)
            df = pd.DataFrame(raw, columns=[f"f{i}" for i in range(raw.shape[1])])
            self.time_values_all = np.arange(len(df))

            df_num = df.copy()
            base_numeric_cols = list(df_num.columns)

            self.feature_cols = base_numeric_cols
            self.target_cols = base_numeric_cols if target_cols is None else list(target_cols)

        # =========================
        # B) 读取 CSV
        # =========================
        else:
            df = pd.read_csv(csv_file_path, dtype=str)

            # 0) 删除无效列
            drop_unnamed = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
            if drop_unnamed:
                df = df.drop(columns=drop_unnamed)

            # 1) 自动识别时间列
            if time_col is None:
                cand = [
                    c for c in df.columns
                    if ("date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower())
                ]
                time_col = cand[0] if cand else None

            engineered_time_cols: list[str] = []

            # 2) 时间解析和排序
            if time_col is not None and time_col in df.columns:
                dt = pd.to_datetime(df[time_col], errors="coerce")

                if dt.notna().any():
                    self.time_col = time_col
                    df[time_col] = dt
                    df = df.dropna(subset=[time_col])

                    if sort_by_time:
                        df = df.sort_values(time_col).reset_index(drop=True)

                    self.time_values_all = df[time_col].to_numpy()
                else:
                    self.time_col = None
                    self.time_values_all = np.arange(len(df))
            else:
                self.time_col = None
                self.time_values_all = np.arange(len(df))

            # 3) 原始数值列识别
            candidate_cols = [c for c in df.columns if c != self.time_col]
            tmp_num = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
            base_numeric_cols = [c for c in candidate_cols if tmp_num[c].notna().any()]

            # 4) 时间特征工程
            if add_time_features and self.time_col is not None:
                dt = df[self.time_col].dt

                df["hour"] = dt.hour.astype(np.float32)
                df["minute"] = dt.minute.astype(np.float32)
                df["dayofweek"] = dt.dayofweek.astype(np.float32)
                df["day"] = dt.day.astype(np.float32)
                df["month"] = dt.month.astype(np.float32)
                df["dayofyear"] = dt.dayofyear.astype(np.float32)

                engineered_time_cols += ["hour", "minute", "dayofweek", "day", "month", "dayofyear"]

                if add_cyc_features:
                    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
                    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)

                    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
                    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)

                    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
                    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

                    engineered_time_cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

            # 5) feature_cols
            if feature_cols is None:
                auto_cols = list(base_numeric_cols)
                for c in engineered_time_cols:
                    if c in df.columns:
                        auto_cols.append(c)

                if len(auto_cols) == 0:
                    auto_cols = [c for c in df.columns if c != self.time_col]

                self.feature_cols = auto_cols
            else:
                self.feature_cols = list(feature_cols)
                for c in self.feature_cols:
                    if c not in df.columns:
                        df[c] = 0.0

            # 6) target_cols
            if target_cols is None:
                if len(base_numeric_cols) > 0:
                    self.target_cols = list(base_numeric_cols)
                else:
                    drop_set = set(engineered_time_cols)
                    self.target_cols = [c for c in self.feature_cols if c not in drop_set]
            else:
                self.target_cols = list(target_cols)

            # 7) 数值化 + 清洗
            df_num = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
            df_num = df_num.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        # =========================
        # C) 数组化
        # =========================
        self.raw_data = df_num.values.astype(np.float32)
        self.full_num_channels = int(self.raw_data.shape[1])

        self.target_feature_indices = [self.feature_cols.index(c) for c in self.target_cols if c in self.feature_cols]
        if len(self.target_feature_indices) == 0:
            self.target_feature_indices = list(range(self.full_num_channels))
            self.target_cols = [self.feature_cols[i] for i in self.target_feature_indices]

        target_set = set(self.target_cols)
        self.known_feature_indices = [i for i, c in enumerate(self.feature_cols) if c not in target_set]
        self.known_feature_cols = [self.feature_cols[i] for i in self.known_feature_indices]

        # =========================
        # D) 划分 train/val/test
        # =========================
        total_len = len(self.raw_data)
        if self.split_borders is not None:
            if len(self.split_borders) not in {2, 3}:
                raise ValueError(
                    "split_borders must be [train_end, val_end] or [train_end, val_end, test_end], "
                    f"got {self.split_borders}"
                )
            train_end = int(self.split_borders[0])
            val_end = int(self.split_borders[1])
            test_end = int(self.split_borders[2]) if len(self.split_borders) == 3 else total_len
        else:
            ratios = self.split_ratios
            if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError(f"split_ratios must have 3 values summing to 1, got {ratios}")
            train_end = int(total_len * ratios[0])
            val_end = int(total_len * (ratios[0] + ratios[1]))
            test_end = total_len

        train_end = max(1, min(int(train_end), total_len))
        val_end = max(train_end, min(int(val_end), total_len))
        test_end = max(val_end, min(int(test_end), total_len))
        self.split_borders = [train_end, val_end, test_end]

        if mode == "train":
            data_slice = self.raw_data[:train_end]
            self.time_slice = self.time_values_all[:train_end] if self.time_values_all is not None else None
        elif mode == "val":
            data_slice = self.raw_data[train_end:val_end]
            self.time_slice = self.time_values_all[train_end:val_end] if self.time_values_all is not None else None
        elif mode == "test":
            data_slice = self.raw_data[val_end:test_end]
            self.time_slice = self.time_values_all[val_end:test_end] if self.time_values_all is not None else None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # =========================
        # E) 归一化
        # =========================
        if scaler is None:
            self.scaler = MinMaxScaler() if self.scaler_type == "minmax" else StandardScaler()
            self.scaler.fit(self.raw_data[:train_end])
        else:
            self.scaler = scaler

        normalized = self.scaler.transform(data_slice).astype(np.float32)
        self.full_data_tensor = torch.from_numpy(normalized)
        if self.target_only_diffusion:
            self.data_tensor = self.full_data_tensor[:, self.target_feature_indices].contiguous()
            if len(self.known_feature_indices) > 0:
                self.known_data_tensor = self.full_data_tensor[:, self.known_feature_indices].contiguous()
            else:
                self.known_data_tensor = self.full_data_tensor.new_zeros((self.full_data_tensor.shape[0], 0))
            self.model_feature_cols = list(self.target_cols)
            self.target_indices = list(range(len(self.target_cols)))
            self.num_channels = int(self.data_tensor.shape[1])
            self.known_future_channels = int(self.known_data_tensor.shape[1])
        else:
            self.data_tensor = self.full_data_tensor
            self.known_data_tensor = self.full_data_tensor.new_zeros((self.full_data_tensor.shape[0], 0))
            self.model_feature_cols = list(self.feature_cols)
            self.target_indices = list(self.target_feature_indices)
            self.num_channels = self.full_num_channels
            self.known_future_channels = 0

        # =========================
        # F) 可选：读取窗口标签
        # =========================
        expected_samples = max(0, len(self.data_tensor) - self.context_len - self.pred_len + 1)

        if label_path is not None and os.path.exists(label_path):
            labels = np.load(label_path)
            if labels.ndim not in (1, 2):
                raise RuntimeError(f"label file should be 1D or 2D, got {labels.shape}")
            if len(labels) != expected_samples:
                raise RuntimeError(
                    f"label count mismatch: got {len(labels)}, expected {expected_samples}, file={label_path}"
                )
            if labels.ndim == 2 and labels.shape[1] != len(self.target_indices):
                raise RuntimeError(
                    f"target-wise label dim mismatch: got {labels.shape[1]}, "
                    f"expected {len(self.target_indices)}, file={label_path}"
                )
            self.sample_labels = labels.astype(np.int64)
            print(f"✅ [Dataset] loaded labels from: {label_path}")
        else:
            self.sample_labels = None
            if label_path is not None:
                print(f"⚠️ [Dataset] label_path not found: {label_path} -> fallback to learned coarse gate")

        if mode == "train":
            channel_info = f"{self.num_channels}"
            if self.target_only_diffusion:
                channel_info = f"{channel_info} (full={self.full_num_channels})"
            print(f"✅ [Dataset] time_col={'None' if self.time_col is None else self.time_col} | channels={channel_info}")
            print(f"✅ [Dataset] feature_cols example: {self.feature_cols[:8]} ...")
            print(f"✅ [Dataset] target_cols: {self.target_cols[:8]} ...")
            if self.target_only_diffusion:
                print(f"✅ [Dataset] known_future_cols: {self.known_feature_cols[:8]} ...")

    def __len__(self):
        return max(0, len(self.data_tensor) - self.context_len - self.pred_len + 1)

    def __getitem__(self, idx):
        idx = int(idx)
        s = idx
        m = idx + self.context_len
        e = idx + self.context_len + self.pred_len

        past = self.data_tensor[s:m, :]
        future = self.data_tensor[m:e, :]
        known_future = self.known_data_tensor[m:e, :]

        if self.channels_first:
            past = past.T.contiguous()
            future = future.T.contiguous()
            known_future = known_future.T.contiguous()
        else:
            past = past.contiguous()
            future = future.contiguous()
            known_future = known_future.contiguous()

        if self.target_only_diffusion:
            sample = {
                "past": past,
                "future": future,
                "known_future": known_future,
            }
            if self.sample_labels is not None:
                sample["coarse_label"] = torch.as_tensor(self.sample_labels[idx], dtype=torch.long)
            return sample

        if self.sample_labels is not None:
            label = torch.as_tensor(self.sample_labels[idx], dtype=torch.long)
            return past, future, label

        return past, future

    def get_time_window(self, idx):
        s = int(idx)
        m = s + self.context_len
        e = s + self.context_len + self.pred_len

        if self.time_slice is None:
            return np.arange(s, m), np.arange(m, e)

        return self.time_slice[s:m], self.time_slice[m:e]

    def _inverse_transform_subset(self, data: np.ndarray, feature_indices: list[int]):
        feature_indices = [int(i) for i in feature_indices]
        if len(feature_indices) == 0:
            return data
        flat = data.reshape(-1, len(feature_indices)).astype(np.float32, copy=False)
        scale_vals = np.asarray(self.scaler.scale_, dtype=np.float32)[feature_indices]
        safe_scale = np.where(np.abs(scale_vals) < 1e-12, 1.0, scale_vals)
        if hasattr(self.scaler, "min_"):  # MinMaxScaler: X_scaled = X * scale_ + min_
            offset = np.asarray(self.scaler.min_, dtype=np.float32)[feature_indices]
            raw = (flat - offset.reshape(1, -1)) / safe_scale.reshape(1, -1)
        else:  # StandardScaler: X_scaled = (X - mean_) / scale_
            mean_vals = np.asarray(self.scaler.mean_, dtype=np.float32)[feature_indices]
            raw = flat * safe_scale.reshape(1, -1) + mean_vals.reshape(1, -1)
        return raw.reshape(data.shape)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data)

        def inverse_by_width(arr: np.ndarray, width: int):
            if width == self.full_num_channels:
                return self.scaler.inverse_transform(arr)
            if width == self.num_channels:
                if self.target_only_diffusion:
                    return self._inverse_transform_subset(arr, self.target_feature_indices)
                return self.scaler.inverse_transform(arr)
            if self.target_only_diffusion and self.known_future_channels > 0 and width == self.known_future_channels:
                return self._inverse_transform_subset(arr, self.known_feature_indices)
            return None

        if data.ndim == 2:
            out = inverse_by_width(data, data.shape[1])
            if out is not None:
                return out
            out = inverse_by_width(data.T, data.shape[0])
            if out is not None:
                return out

        if data.ndim == 3:
            x = data[0]
            out = inverse_by_width(x, x.shape[-1])
            if out is not None:
                return out
            out = inverse_by_width(x.T, x.shape[0])
            if out is not None:
                return out

        try:
            flat = data.reshape(-1, self.full_num_channels)
            return self.scaler.inverse_transform(flat)
        except Exception:
            try:
                flat = data.reshape(-1, self.num_channels)
                out = inverse_by_width(flat, self.num_channels)
                if out is not None:
                    return out
            except Exception:
                pass
            return data
