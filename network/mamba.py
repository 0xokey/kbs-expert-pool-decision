from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser, SinusoidalTimeEmbedding
from .s4 import SelectiveSSM


class LagFusionBlock(nn.Module):
    """Lightweight DiM-TS-inspired lag fusion around a selective SSM."""

    def __init__(self, d_model: int, lags: tuple[int, ...] = (1, 24, 48), d_state: int = 64):
        super().__init__()
        self.lags = tuple(int(v) for v in lags if int(v) > 0)
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state)
        self.lag_gate = nn.Sequential(
            nn.Linear(d_model * (1 + len(self.lags)), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.lag_proj = nn.Linear(d_model * (1 + len(self.lags)), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        lagged = [x]
        for lag in self.lags:
            if lag >= x.size(1):
                lagged.append(torch.zeros_like(x))
            else:
                pad = torch.zeros(x.size(0), lag, x.size(2), device=x.device, dtype=x.dtype)
                lagged.append(torch.cat([pad, x[:, :-lag, :]], dim=1))
        fused = torch.cat(lagged, dim=-1)
        candidate = self.lag_proj(fused)
        gate = self.lag_gate(fused)
        x = self.norm(x + gate * candidate)
        return self.ssm(x)


class ChannelGRUBlock(nn.Module):
    """Channel-wise GRU analogue for lightweight cross-variable scanning."""

    def __init__(self, seq_length: int, num_channels: int, hidden_dim: int):
        super().__init__()
        self.seq_length = int(seq_length)
        self.num_channels = int(num_channels)
        self.channel_rnn = nn.GRU(
            input_size=seq_length,
            hidden_size=seq_length,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Linear(seq_length * 2, seq_length)
        self.mix = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        # x_bcl: [B, C, L]
        if x_bcl.size(-1) != self.seq_length:
            x_bcl = F.interpolate(x_bcl, size=self.seq_length, mode="linear", align_corners=False)
        y, _ = self.channel_rnn(x_bcl)
        y = self.out_proj(y)
        return x_bcl + y + self.mix(x_bcl)


ChannelPermutationBlock = ChannelGRUBlock


class DiMTSMambaDiffuser(BaseDiffuser):
    """DiM-TS-inspired Mamba expert for time-series diffusion.

    It approximates two core ideas from DiM-TS: temporal lag fusion and channel
    scanning. This is intentionally lightweight so it can serve as one MoE
    expert without dominating the expert pool.
    """
    diffusion_profile = {
        "source_model": "DiM-TS/Mamba-style diffusion expert",
        "backbone_family": "mamba_ssm",
        "preserved_operations": [
            "diffusion timestep embedding",
            "future/history concatenation conditioning",
            "multi-lag temporal fusion",
            "selective state-space temporal blocks",
            "channel-wise recurrent scanning",
            "noise prediction head",
        ],
        "limitations": [
            "Uses the shared MoE diffusion scheduler; per-expert Mamba diffusion sampling is not yet separated.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 2,
        lags: tuple[int, ...] = (1, 24, 48),
    ):
        super().__init__(seq_length, input_channels)
        self.in_proj = nn.Linear(input_channels * 2, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_blocks = nn.ModuleList(
            [LagFusionBlock(d_model=d_model, lags=lags, d_state=d_state) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.to_channels = nn.Linear(d_model, input_channels)
        self.channel_block = ChannelGRUBlock(seq_length, input_channels, d_model)
        self.out_proj = nn.Conv1d(input_channels, input_channels, kernel_size=1)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1).transpose(1, 2)
        h = self.in_proj(x)
        h = h + self.time_mlp(self.time_emb(t)).unsqueeze(1)

        for block in self.temporal_blocks:
            h = block(h)

        y = self.to_channels(self.out_norm(h)).transpose(1, 2).contiguous()
        y = self.channel_block(y)
        return self.out_proj(y)
