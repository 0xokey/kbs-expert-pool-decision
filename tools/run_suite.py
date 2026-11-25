"""Sequential multi-seed runner for reproducible experiment suites.

Examples
--------
python tools/run_suite.py --config configs/moe_hierarchical_v1.json --seeds 42 43 44 --tag main
python tools/run_suite.py --config configs/moe_hierarchical_v1.json --seeds 42 --tag abl_no_prior \
    --set channel_prior_alpha=0.0 --set lambda_channel_prior=0.0

Each run is a normal `train.py` invocation; stdout/stderr is tee'd to
`<results>/_suite_logs/<tag>__seed<seed>.log`. Completed runs are detected by
`model_weights.pth` + `evaluation/Final/metrics/summary.json`, so the suite is
resumable and safe to relaunch after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def find_completed_run(results_dir: Path, tag: str, seed: int):
    """Return an existing finished run directory for (tag, seed), if any."""
    if not results_dir.exists():
        return None
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        config = _read_json(run_dir / "config.json")
        if str(config.get("run_tag", "")) != str(tag):
            continue
        if int(config.get("seed", -1)) != int(seed):
            continue
        done = (run_dir / "model_weights.pth").exists() and (
            run_dir / "evaluation" / "Final" / "metrics" / "summary.json"
        ).exists()
        if done:
            return run_dir
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a config across multiple seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--tag", default=None, help="run_tag shared by all seeds.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    base_tag = args.tag or json.loads(config_path.read_text(encoding="utf-8")).get("run_tag") or config_path.stem
    results_dir = Path(args.results_dir)
    log_dir = results_dir / "_suite_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    started = time.time()
    for seed in args.seeds:
        # The run_tag stays identical across seeds so the aggregator can group
        # them into one mean±std row; the seed itself disambiguates the runs.
        tag = str(base_tag)
        existing = find_completed_run(results_dir, tag, seed) if args.skip_existing else None
        if existing is not None:
            print(f"[Suite] skip seed={seed} (completed: {existing.name})")
            continue

        cmd = [
            args.python,
            str(ROOT / "train.py"),
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--run_tag",
            tag,
            "--results_dir",
            str(results_dir),
        ]
        for item in args.set:
            cmd.extend(["--set", item])

        log_path = log_dir / f"{tag}__seed{seed}.log"
        print(f"[Suite] seed={seed} tag={tag} -> {log_path}")
        print("[Suite] " + " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
            returncode = process.wait()
        elapsed = time.time() - t0
        if returncode != 0:
            failures.append((seed, returncode))
            print(f"[Suite] seed={seed} FAILED (exit={returncode}) after {elapsed / 60:.1f} min")
        else:
            print(f"[Suite] seed={seed} done in {elapsed / 60:.1f} min")

    total = (time.time() - started) / 60.0
    print(f"[Suite] finished {len(args.seeds)} seed(s) in {total:.1f} min; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
