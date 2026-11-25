from .base import BaseDiffuser
from .ddpm_unet import UNetDiffuser
from .diffwave import DiffWaveDiffuser
from .patch_transformer import PatchTransformerDiffuser
from .diffstg_ugnet import UGNetDiffuser
from .csdi import CSDIDiffuser, CSDIAttentionDiffuser, SpectralDiffuser
from .s4 import S4Diffuser, SelectiveSSMDiffuser
from .mamba import DiMTSMambaDiffuser
from .rnn import RNNDiffuser
from .linear_expert import DLinearDiffuser
from .patchtst_expert import PatchTSTDiffuser
from .itransformer_expert import ITransformerLiteDiffuser

DEFAULT_EXPERTS = [
    "UNetDiffuser",
    "DiffWaveDiffuser",
    "PatchTransformerDiffuser",
    "UGNetDiffuser",
    "CSDIAttentionDiffuser",
    "SpectralDiffuser",
    "SelectiveSSMDiffuser",
    "DiMTSMambaDiffuser",
    "RNNDiffuser",
]

MODEL_REGISTRY = {
    "UNetDiffuser": UNetDiffuser,
    "DiffWaveDiffuser": DiffWaveDiffuser,
    "PatchTransformerDiffuser": PatchTransformerDiffuser,
    "UGNetDiffuser": UGNetDiffuser,
    "CSDIDiffuser": CSDIDiffuser,
    "CSDIAttentionDiffuser": CSDIAttentionDiffuser,
    "SpectralDiffuser": SpectralDiffuser,
    "S4Diffuser": S4Diffuser,
    "SelectiveSSMDiffuser": SelectiveSSMDiffuser,
    "DiMTSMambaDiffuser": DiMTSMambaDiffuser,
    "RNNDiffuser": RNNDiffuser,
    "DLinearDiffuser": DLinearDiffuser,
    "PatchTSTDiffuser": PatchTSTDiffuser,
    "ITransformerLiteDiffuser": ITransformerLiteDiffuser,
}

__all__ = [
    "BaseDiffuser",
    "UNetDiffuser",
    "DiffWaveDiffuser",
    "PatchTransformerDiffuser",
    "UGNetDiffuser",
    "CSDIDiffuser",
    "CSDIAttentionDiffuser",
    "SpectralDiffuser",
    "S4Diffuser",
    "SelectiveSSMDiffuser",
    "DiMTSMambaDiffuser",
    "RNNDiffuser",
    "DLinearDiffuser",
    "PatchTSTDiffuser",
    "ITransformerLiteDiffuser",
    "MODEL_REGISTRY",
    "DEFAULT_EXPERTS",
]
