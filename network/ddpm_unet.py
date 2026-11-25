from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock1D(in_ch, out_ch)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        d = self.down(h)
        return h, d


class UpBlock1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock1D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDiffuser(BaseDiffuser):
    diffusion_profile = {
        "source_model": "DDPM UNet-style denoiser",
        "backbone_family": "unet",
        "preserved_operations": [
            "diffusion timestep embedding",
            "history-conditioned noisy future input",
            "downsample/upsample UNet path",
            "skip connections",
            "noise prediction head",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        hidden_dim: int = 64,
        time_dim: int = 64,
    ):
        super().__init__(seq_length, input_channels)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.in_proj = nn.Conv1d(input_channels * 2, hidden_dim, kernel_size=1)

        self.down1 = DownBlock1D(hidden_dim, hidden_dim)
        self.down2 = DownBlock1D(hidden_dim, hidden_dim * 2)

        self.mid = ConvBlock1D(hidden_dim * 2, hidden_dim * 2)

        self.up1 = UpBlock1D(hidden_dim * 2, hidden_dim * 2, hidden_dim)
        self.up2 = UpBlock1D(hidden_dim, hidden_dim, hidden_dim)

        self.out_proj = nn.Conv1d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1)
        x = self.in_proj(x)

        t_emb = self.time_mlp(self.time_emb(t)).unsqueeze(-1)
        x = x + t_emb

        s1, d1 = self.down1(x)
        s2, d2 = self.down2(d1)

        mid = self.mid(d2)
        u1 = self.up1(mid, s2)
        u2 = self.up2(u1, s1)

        return self.out_proj(u2)
