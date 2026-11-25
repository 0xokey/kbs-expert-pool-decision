from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels: int, dilation: int, cond_channels: int):
        super().__init__()
        self.dilated = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.cond_proj = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1)
        self.diff_proj = nn.Linear(residual_channels, residual_channels)
        self.out_proj = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, diffusion_step: torch.Tensor, cond: torch.Tensor):
        y = x + self.diff_proj(diffusion_step).unsqueeze(-1)
        y = self.dilated(y) + self.cond_proj(cond)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        y = self.out_proj(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWaveDiffuser(BaseDiffuser):
    diffusion_profile = {
        "source_model": "DiffWave-style diffusion denoiser",
        "backbone_family": "diffwave",
        "preserved_operations": [
            "diffusion timestep embedding",
            "dilated residual convolution stack",
            "conditioning projection from history",
            "skip aggregation",
            "noise prediction head",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        residual_channels: int = 64,
        residual_layers: int = 8,
        dilation_cycle_length: int = 4,
    ):
        super().__init__(seq_length, input_channels)

        self.input_proj = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.cond_proj = nn.Conv1d(input_channels, residual_channels, kernel_size=1)

        self.time_emb = SinusoidalTimeEmbedding(residual_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(residual_channels, residual_channels),
            nn.SiLU(),
            nn.Linear(residual_channels, residual_channels),
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    cond_channels=residual_channels,
                )
                for i in range(residual_layers)
            ]
        )

        self.skip_proj = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, input_channels, kernel_size=1),
        )

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = self.input_proj(noisy_future)
        cond = self.cond_proj(past)
        t_emb = self.time_mlp(self.time_emb(t))

        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, t_emb, cond)
            skips.append(skip)

        x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(skips))
        x = self.skip_proj(x)
        return self.out_proj(x)
