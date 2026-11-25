import inspect
import os

import torch


def instantiate_model(ModelClass, seq_len: int, num_channels: int, device: torch.device, extra_kwargs=None):
    sig = inspect.signature(ModelClass.__init__)
    params = sig.parameters

    kwargs = {}

    if "seq_length" in params:
        kwargs["seq_length"] = int(seq_len)
    elif "seq_len" in params:
        kwargs["seq_len"] = int(seq_len)

    if "input_channels" in params:
        kwargs["input_channels"] = int(num_channels)
    elif "num_channels" in params:
        kwargs["num_channels"] = int(num_channels)
    elif "in_channels" in params:
        kwargs["in_channels"] = int(num_channels)

    if extra_kwargs:
        kwargs.update({k: v for k, v in dict(extra_kwargs).items() if k in params})

    return ModelClass(**kwargs).to(device)


def resolve_context_pred_lengths(config: dict) -> tuple[int, int]:
    seq_len = int(config.get("seq_len", 168))
    context_len = int(config.get("context_len", seq_len))
    pred_len = int(config.get("pred_len", seq_len))
    return context_len, pred_len


def resolve_label_path(config, csv_path: str | None, mode: str):
    if csv_path is None:
        return None

    label_dir = config.get("label_dir", None)
    if not label_dir:
        return None

    base = os.path.splitext(os.path.basename(csv_path))[0]
    path = os.path.join(label_dir, f"{base}_{mode}_labels.npy")
    return path if os.path.exists(path) else None
