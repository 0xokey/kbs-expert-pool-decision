"""Aggregate hard-channel run artifacts into a single, paper-friendly report.

The reporter is intentionally read-only with respect to training artifacts: it
re-validates every run with :mod:`tools.validate_hard_channel_run` and then
collects validation/test metrics from `evaluation/<split>/metrics/summary.json`.

Usage:
    python tools/report_hard_channel_runs.py
    python tools/report_hard_channel_runs.py --results-dir results --validated-only
    python tools/report_hard_channel_runs.py --refresh-summary --include-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # keep CJK/emoji output working on GBK consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.validate_hard_channel_run import (  # noqa: E402
    _failures_for_run,
    _is_hard_channel_run,
    _run_report_row,
    _write_csv,
)


REPORT_FIELDS = [
    "run_name",
    "run_dir",
    "run_tag",
    "seed",
    "config_model",
    "routing_mode",
    "expert_diffusion_profile_policy",
    "target_only_diffusion",
    "hard_final_routing",
    "final_top_k",
    "num_experts",
    "forecast_objective",
    "num_windows",
    "RMSE",
    "MAE",
    "R2",
    "CRPS",
    "status",
    "validation_status",
    "num_failures",
    "failures",
    "test_num_windows",
    "test_RMSE",
    "test_MAE",
    "test_R2",
    "test_CRPS",
    "test_status",
]

SUMMARY_FIELDS = REPORT_FIELDS


def _read_json(path: Path):
    """Read a JSON object; missing or malformed files yield an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_validated(row) -> bool:
    """Return True when a report row carries no validation failures.

    Legacy summaries (written before `validation_status` existed) only have
    `validation_failures`; an empty string means the run passed.
    """
    status = str(row.get("validation_status", "") or "").strip().upper()
    if status:
        return status == "OK"
    return str(row.get("validation_failures", "") or "").strip() == ""


def _should_refresh_summary(args, out_dir: Path) -> bool:
    """`--validated-only` or `--refresh-summary` force a summary rebuild."""
    out_dir = Path(out_dir)
    if bool(getattr(args, "refresh_summary", False)):
        return True
    if bool(getattr(args, "validated_only", False)):
        return True
    return not (out_dir / "run_summary.csv").exists()


def _split_metrics(run_dir: Path, split: str) -> dict:
    summary = _read_json(run_dir / "evaluation" / split / "metrics" / "summary.json")
    manifest_path = run_dir / "evaluation" / split / "metrics" / "metrics_manifest.json"
    if not summary and not manifest_path.exists():
        return {}
    return summary


def _report_row(run_dir: Path) -> dict:
    failures = _failures_for_run(run_dir)
    row = _run_report_row(run_dir, failures)
    config = _read_json(run_dir / "config.json")
    val = _split_metrics(run_dir, "Final")
    test = _split_metrics(run_dir, "FinalTest")

    row.update(
        {
            "seed": config.get("seed", ""),
            "MAE": val.get("MAE", ""),
            "CRPS": val.get("CRPS", ""),
            "validation_status": row.get("status", ""),
            "validation_failures": row.get("failures", ""),
            "test_num_windows": test.get("num_windows", ""),
            "test_RMSE": test.get("RMSE", ""),
            "test_MAE": test.get("MAE", ""),
            "test_R2": test.get("R2", ""),
            "test_CRPS": test.get("CRPS", ""),
            "test_status": ("OK" if test else "missing"),
        }
    )
    return row


def collect_rows(results_dir: Path, *, validated_only: bool = False):
    rows = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir() or not _is_hard_channel_run(run_dir):
            continue
        row = _report_row(run_dir)
        if validated_only and not _is_validated(row):
            continue
        rows.append(row)
    return rows


def _print_table(rows):
    if not rows:
        print("[Report] no hard-channel runs found")
        return
    header = f"{'run_name':<52} {'val_RMSE':>9} {'val_R2':>8} {'test_RMSE':>10} {'test_R2':>8} {'status':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        def _fmt(value, digits=4):
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "-"
        print(
            f"{str(row['run_name'])[:52]:<52} "
            f"{_fmt(row.get('RMSE')):>9} {_fmt(row.get('R2'), 2):>8} "
            f"{_fmt(row.get('test_RMSE')):>10} {_fmt(row.get('test_R2'), 2):>8} "
            f"{str(row.get('status')):>6}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate hard-channel run artifacts into a CSV report.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="Report CSV path (default: <results>/hard_channel_report.csv).")
    parser.add_argument("--summary-out", default=None, help="Summary CSV path (default: <results>/run_summary.csv).")
    parser.add_argument("--refresh-summary", action="store_true")
    parser.add_argument("--validated-only", action="store_true")
    parser.add_argument("--include-test", action="store_true", help="Keep test-split columns (default: on).")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    rows = collect_rows(results_dir, validated_only=bool(args.validated_only))
    out_path = Path(args.out) if args.out else results_dir / "hard_channel_report.csv"
    _write_csv(out_path, rows, REPORT_FIELDS)
    _print_table(rows)

    summary_path = Path(args.summary_out) if args.summary_out else results_dir / "run_summary.csv"
    if _should_refresh_summary(args, summary_path.parent):
        _write_csv(summary_path, rows, SUMMARY_FIELDS)
        print(f"[Report] summary  -> {summary_path}")
    print(f"[Report] report   -> {out_path} ({len(rows)} runs)")
    failed = [row for row in rows if not _is_validated(row)]
    if failed:
        print(f"[Report] {len(failed)} run(s) with validation failures")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
