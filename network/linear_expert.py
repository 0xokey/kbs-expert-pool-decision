"""DLinear-style linear expert for the heterogeneous MoE.

DLinear (Zeng et al., 2023) is a strong, cheap forecasting baseline. Adding it
as one expert lets the channel router fall back to a linear map when a diffusion
denoiser is not worth its variance. With `prediction_type=x0` the expert is
trained with exactly the DLinear objective (MSE against the clean residual
target), and at inference it ignores the diffusion timestep.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseDiffuser


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        x = torch.cat([front, x, end], dim=-1)
        return self.avg(x)


class DLinearDiffuser(BaseDiffuser):
    # The linear map is most useful on the full lookback window, so this expert
    # opts into the raw context instead of the compressed 96-step context.
    uses_full_context = True

    diffusion_profile = {
        "source_model": "DLinear (linear decomposition) expert",
        "backbone_family": "linear",
        "preserved_operations": [
            "history-conditioned input",
            "series decomposition into trend/seasonal",
            "direct multi-horizon linear map",
        ],
        "limitations": [
            "Deterministic: ignores the diffusion timestep, used as a routed expert.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        context_length: int | None = None,
        kernel_size: int = 25,
    ):
        super().__init__(seq_length, input_channels)
        self.context_length = int(context_length or seq_length)
        self.decomposition = MovingAverage(kernel_size)
        self.linear_seasonal = nn.Linear(self.context_length, int(seq_length))
        self.linear_trend = nn.Linear(self.context_length, int(seq_length))

    def forward(self, noisy_future, t, past):
        if past.size(-1) != self.context_length:
            past = torch.nn.functional.interpolate(
                past, size=self.context_length, mode="linear", align_corners=False
            )
        trend = self.decomposition(past)
        seasonal = past - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)
