from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.transpose(1, 2)


class PatchTransformerDiffuser(BaseDiffuser):
    diffusion_profile = {
        "source_model": "Patch Transformer diffusion denoiser",
        "backbone_family": "patch_transformer",
        "preserved_operations": [
            "diffusion timestep embedding",
            "patch embedding over future/history channels",
            "Transformer encoder over temporal patches",
            "patch reconstruction head",
            "noise prediction head",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        patch_len: int = 12,
        stride: int = 8,
    ):
        super().__init__(seq_length, input_channels)

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(input_channels * 2, d_model, patch_len, stride)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_channels * patch_len)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1)
        tokens = self.patch_embed(x)

        t_emb = self.time_emb(t).unsqueeze(1)
        tokens = tokens + t_emb

        z = self.encoder(tokens)
        z = self.norm(z)
        patch_out = self.head(z)

        B, N, _ = patch_out.shape
        C = self.input_channels
        patch_out = patch_out.view(B, N, C, self.patch_len)

        L_out = N * self.patch_len
        y = patch_out.permute(0, 2, 1, 3).reshape(B, C, L_out)

        if L_out != self.seq_length:
            y = F.interpolate(y, size=self.seq_length, mode="linear", align_corners=False)

        return y
