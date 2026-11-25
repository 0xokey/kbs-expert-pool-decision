from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_methods import (
    ChannelProtocolMoE,
    FULL_PROTOCOL_REGISTRY,
    FullProtocolBatch,
    StrictReproductionUnavailableError,
)


def _full_kwargs_for(name: str) -> dict:
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require_all", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    context_len = 24
    pred_len = 24
    batch = FullProtocolBatch(
        past=torch.randn(2, 2, context_len, device=device),
        future=torch.randn(2, 2, pred_len, device=device),
    )

    runnable = []
    skipped = []
    for name, cls in FULL_PROTOCOL_REGISTRY.items():
        kwargs = _full_kwargs_for(name)
        print(f"{name}: checking full default protocol...", flush=True)
        try:
            protocol = cls(target_dim=2, context_len=context_len, pred_len=pred_len, device=device, **kwargs)
        except StrictReproductionUnavailableError as exc:
            skipped.append((name, str(exc)))
            print(f"{name}: SKIP strict reproduction unavailable -> {exc}")
            continue
        except (ImportError, RuntimeError) as exc:
            if "Strict SSD-TS requires exact BiSSM2Imputer" not in str(exc) and isinstance(exc, RuntimeError):
                raise
            skipped.append((name, str(exc)))
            print(f"{name}: SKIP strict reproduction unavailable -> {exc}")
            continue
        loss = protocol.training_loss(batch)
        sample = protocol.sample_channel(batch, channel_idx=0, n_samples=1)
        print(f"{name}: loss={float(loss.detach().cpu()):.6f} sample_shape={tuple(sample.shape)}")
        runnable.append(name)

    if args.require_all and skipped:
        unavailable = [name for name, _ in skipped]
        raise SystemExit(f"Strict smoke has unavailable methods: {unavailable}")

    if not runnable:
        raise SystemExit("No runnable strict protocols are available.")

    channel_batch = FullProtocolBatch(
        past=torch.randn(2, 7, context_len, device=device),
        future=torch.randn(2, 7, pred_len, device=device),
    )
    assignment = {idx: runnable[idx % len(runnable)] for idx in range(7)}
    moe = ChannelProtocolMoE(
        target_dim=7,
        context_len=context_len,
        pred_len=pred_len,
        device=device,
        channel_to_method=assignment,
        protocol_registry=FULL_PROTOCOL_REGISTRY,
        protocol_kwargs={
            method: _full_kwargs_for(method)
            for method in FULL_PROTOCOL_REGISTRY
        },
    )
    loss = moe.training_loss(channel_batch)
    sample = moe.sample(channel_batch, n_samples=1)
    print(f"ChannelProtocolMoE: loss={float(loss.detach().cpu()):.6f} sample_shape={tuple(sample.shape)}")


if __name__ == "__main__":
    main()
