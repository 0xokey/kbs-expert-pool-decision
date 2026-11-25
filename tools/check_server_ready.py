from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_methods import STRICT_REPRODUCTION_STATUS

AVAILABLE_STRICT_METHODS = {
    method
    for method, status in STRICT_REPRODUCTION_STATUS.items()
    if status.get("status") == "available"
}
KNOWN_STRICT_METHODS = set(STRICT_REPRODUCTION_STATUS)


def _selected_methods(config: dict) -> set[str]:
    if bool(config.get("phase0_channel_selection", False)):
        methods = config.get("phase0_candidate_methods", [])
        return {str(method) for method in methods} if isinstance(methods, list) else set()
    channel_methods = config.get("channel_methods", {})
    return {str(method) for method in channel_methods.values()} if isinstance(channel_methods, dict) else set()


def _missing_method_dependencies(methods: set[str]) -> list[str]:
    failures = []
    if "SSD-TS" in methods:
        try:
            import mamba_ssm  # noqa: F401
        except Exception as exc:
            failures.append(f"SSD-TS requires mamba-ssm for exact BiSSM2/Mamba2 reproduction: {exc}")
    return failures


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def _failures(config: dict, *, check_cuda: bool) -> list[str]:
    failures = []
    if str(config.get("device", "")).lower() != "cuda":
        failures.append("config.device must be 'cuda' for server training")
    if not bool(config.get("require_cuda", False)):
        failures.append("config.require_cuda must be true for server training")
    if config.get("debug_max_train_windows", None) is not None:
        failures.append("server config must not set debug_max_train_windows")
    if int(config.get("epochs", 0)) <= 1:
        failures.append("server config epochs should be greater than 1")
    for legacy_key in ("num_steps", "hidden_dim"):
        if legacy_key in config:
            failures.append(f"server config must not set legacy lightweight key {legacy_key!r}")

    csv_path = Path(str(config.get("csv_path", "")))
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.is_file():
        failures.append(f"csv_path does not exist: {csv_path}")

    if bool(config.get("phase0_channel_selection", False)):
        methods = config.get("phase0_candidate_methods", [])
        if not isinstance(methods, list) or not methods:
            failures.append("phase0_candidate_methods must be a non-empty list")
        else:
            for method in methods:
                if method not in KNOWN_STRICT_METHODS:
                    failures.append(f"phase0 candidate unknown strict method {method!r}")
                if method not in AVAILABLE_STRICT_METHODS:
                    failures.append(f"phase0 candidate unavailable strict method {method!r}")
    else:
        channel_methods = config.get("channel_methods", {})
        if not isinstance(channel_methods, dict) or not channel_methods:
            failures.append("config.channel_methods must be a non-empty object when phase0_channel_selection=false")
        else:
            for channel, method in channel_methods.items():
                if method not in KNOWN_STRICT_METHODS:
                    failures.append(f"{channel}: unknown strict method {method!r}")
                if method not in AVAILABLE_STRICT_METHODS:
                    failures.append(f"{channel}: unavailable strict method {method!r}")

    if check_cuda and not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is false")
    failures.extend(_missing_method_dependencies(_selected_methods(config)))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict full-protocol server readiness.")
    parser.add_argument("--config", default="configs/full_protocol_etth1_server.json")
    parser.add_argument(
        "--skip-cuda-runtime-check",
        action="store_true",
        help="Validate config only; do not require torch.cuda.is_available().",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = _load_config(config_path)
    failures = _failures(config, check_cuda=not args.skip_cuda_runtime_check)
    if failures:
        print("[ServerReady] FAILED")
        for item in failures:
            print(f" - {item}")
        raise SystemExit(1)

    print("[ServerReady] OK")
    print(f"config={config_path}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
