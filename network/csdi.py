from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class TemporalFeatureAttentionBlock(nn.Module):
    """CSDI-style temporal attention followed by feature attention.

    CSDI conditions diffusion with observed values and learns both temporal and
    feature dependencies. For forecasting, `past` is the available condition, so
    this block keeps the same two-axis attention idea over [feature, time].
    """

    def __init__(self, hidden_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_attn = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.ff_norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, H, C, L]
        B, H, C, L = h.shape

        temporal_tokens = h.permute(0, 2, 3, 1).reshape(B * C, L, H)
        temporal_out, _ = self.temporal_attn(temporal_tokens, temporal_tokens, temporal_tokens, need_weights=False)
        temporal_tokens = self.temporal_norm(temporal_tokens + temporal_out)
        h = temporal_tokens.reshape(B, C, L, H).permute(0, 3, 1, 2).contiguous()

        feature_tokens = h.permute(0, 3, 2, 1).reshape(B * L, C, H)
        feature_out, _ = self.feature_attn(feature_tokens, feature_tokens, feature_tokens, need_weights=False)
        feature_tokens = self.feature_norm(feature_tokens + feature_out)
        h = feature_tokens.reshape(B, L, C, H).permute(0, 3, 2, 1).contiguous()

        h = self.ff_norm(h + self.ff(h))
        return h


class CSDIAttentionDiffuser(BaseDiffuser):
    """CSDI-like attention denoiser for forecasting.

    This is not a full imputation model with random observed masks. It aligns
    the expert with CSDI's core denoiser idea: condition on observed history and
    model temporal/feature dependencies with two-axis attention.
    """
    diffusion_profile = {
        "source_model": "CSDI-style conditional diffusion denoiser",
        "backbone_family": "temporal_feature_attention",
        "preserved_operations": [
            "diffusion timestep embedding",
            "observed-history conditioning",
            "temporal attention",
            "feature attention",
            "2D projection over feature/time axes",
            "noise prediction head",
        ],
        "limitations": [
            "Forecasting adaptation; does not use CSDI random imputation masks.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(seq_length, input_channels)
        self.in_proj = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [TemporalFeatureAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.stack([noisy_future, past], dim=1)  # [B, 2, C, L]
        h = self.in_proj(x)
        h = h + self.time_mlp(self.time_emb(t)).view(t.size(0), -1, 1, 1)

        for block in self.blocks:
            h = block(h)

        return self.out_proj(h).squeeze(1).contiguous()


class FNOSpectralBlock(nn.Module):
    """Fourier Neural Operator block: spectral conv plus local residual."""

    def __init__(self, channels: int, modes: int = 16):
        super().__init__()
        self.modes = modes
        self.scale = nn.Parameter(torch.randn(channels, modes) * 0.02)
        self.phase = nn.Parameter(torch.randn(channels, modes) * 0.02)
        self.local = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_dtype = x.dtype
        x_f32 = x.float()
        _, _, L = x_f32.shape
        x_ft = torch.fft.rfft(x_f32, n=L, dim=-1)
        modes = min(self.modes, x_ft.size(-1))

        w = (
            1.0
            + self.scale[:, :modes].float()
            + 1j * self.phase[:, :modes].float()
        ).unsqueeze(0)
        x_ft_mod = torch.cat([x_ft[:, :, :modes] * w, x_ft[:, :, modes:]], dim=-1)
        y = torch.fft.irfft(x_ft_mod, n=L, dim=-1)
        return y.to(dtype=inp_dtype) + self.local(x)


class SpectralDiffuser(BaseDiffuser):
    """Frequency-domain denoiser expert for periodic patterns."""
    diffusion_profile = {
        "source_model": "FNO/spectral diffusion denoiser",
        "backbone_family": "spectral_fno",
        "preserved_operations": [
            "diffusion timestep embedding",
            "history-conditioned noisy future input",
            "Fourier-domain mode modulation",
            "local residual correction",
            "noise prediction head",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        hidden_dim: int = 64,
        fno_modes: int = 16,
        num_blocks: int = 3,
    ):
        super().__init__(seq_length, input_channels)
        self.in_proj = nn.Conv1d(input_channels * 2, hidden_dim, kernel_size=1)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FNOSpectralBlock(hidden_dim, modes=fno_modes) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv1d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1)
        h = self.in_proj(x)
        h = h + self.time_mlp(self.time_emb(t)).unsqueeze(-1)

        for block in self.blocks:
            h = F.silu(h + block(h))

        return self.out_proj(h)


class CSDIDiffuser(SpectralDiffuser):
    """Legacy alias kept for old checkpoints.

    Older experiments used the name `CSDIDiffuser` for the FNO-lite spectral
    expert. New configs should use `CSDIAttentionDiffuser` for CSDI-like
    temporal/feature attention, or `SpectralDiffuser` for the frequency expert.
    """

    pass
