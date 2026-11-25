from __future__ import annotations

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


FULL_DIFFUSION_PROTOCOL_OPERATIONS = [
    "forward_noise_process",
    "timestep_sampling",
    "denoising_network_forward",
    "training_loss",
    "reverse_sampling_step",
    "conditioning_adapter",
]


DEFAULT_DIFFUSION_PROFILE = {
    "protocol_version": 1,
    "current_integration_level": "denoiser_extracted",
    "target_integration_level": "full_diffusion_expert",
    "routing_role": "hard_channel_selected_expert",
    "training_objective": "noise_prediction",
    "scheduler_scope": "shared_global",
    "noise_schedule": "linear",
    "num_timesteps": 1000,
    "sampling": {
        "sampler": "ddim",
        "ddim_t_start": 800,
        "eta": 0.0,
    },
    "preserved_operations": [],
    "full_protocol_operations": [],
    "required_full_protocol_operations": FULL_DIFFUSION_PROTOCOL_OPERATIONS,
    "limitations": [
        "This expert is used as a denoiser inside the shared MoE diffusion scheduler.",
    ],
}


def _merge_profile(base: dict, override: dict | None) -> dict:
    out = {}
    for key, value in base.items():
        if isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


class ChannelMixBlock(nn.Module):
    """Per-step cross-channel mixing (1x1 conv, zero-init residual).

    Gives every expert the ability to exchange information across channels
    at each time step before its main computation.  Zero-initialised so the
    residual path starts as identity and the expert body still controls
    the final output.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        hidden = max(int(num_channels * 2), 8)
        self.mix = nn.Sequential(
            nn.Conv1d(num_channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, num_channels, kernel_size=1),
        )
        nn.init.xavier_uniform_(self.mix[0].weight)
        nn.init.zeros_(self.mix[0].bias)
        nn.init.zeros_(self.mix[2].weight)
        nn.init.zeros_(self.mix[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(x)


class BaseDiffuser(nn.Module, ABC):
    diffusion_profile: dict = {}

    def __init__(self, seq_length: int, input_channels: int):
        super().__init__()
        self.seq_length = int(seq_length)
        self.input_channels = int(input_channels)
        self.channel_mix = ChannelMixBlock(input_channels)

    def get_diffusion_profile(self) -> dict:
        profile = _merge_profile(DEFAULT_DIFFUSION_PROFILE, getattr(self, "diffusion_profile", {}))
        preserved = set(str(item) for item in profile.get("preserved_operations", []))
        full_protocol = set(str(item) for item in profile.get("full_protocol_operations", []))
        required = [str(item) for item in profile.get("required_full_protocol_operations", [])]
        profile["missing_full_protocol_operations"] = [
            item for item in required if item not in preserved and item not in full_protocol
        ]
        profile["class_name"] = type(self).__name__
        profile["seq_length"] = int(self.seq_length)
        profile["input_channels"] = int(self.input_channels)
        return profile

    @staticmethod
    def _normalize_inputs(
        noisy_future: torch.Tensor | None,
        t: torch.Tensor | None,
        past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past is None:
            raise ValueError("past cannot be None")

        if noisy_future is None:
            noisy_future = torch.zeros_like(past)

        if t is None:
            t = torch.zeros(past.size(0), dtype=torch.long, device=past.device)

        if noisy_future.dim() != 3 or past.dim() != 3:
            raise RuntimeError(
                f"expected [B,C,L], got noisy_future={tuple(noisy_future.shape)}, past={tuple(past.shape)}"
            )

        if past.size(-1) != noisy_future.size(-1):
            past = F.interpolate(past, size=noisy_future.size(-1), mode="linear", align_corners=False)

        return noisy_future, t

    @abstractmethod
    def forward(
        self,
        noisy_future: torch.Tensor | None,
        t: torch.Tensor | None,
        past: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            t = t.view(-1)

        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return t.float().unsqueeze(1)

        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if emb.size(1) < self.dim:
            pad = emb.new_zeros(emb.size(0), self.dim - emb.size(1))
            emb = torch.cat([emb, pad], dim=1)
        return emb
