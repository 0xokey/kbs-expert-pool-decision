"""Aggregate routing_analysis_summary.csv into per-dataset mean/std."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def dataset_key(run_name: str) -> str:
    name = run_name.lower()
    for key, label in (
        ("etth1", "ETTh1"), ("ettm1", "ETTm1"), ("weather", "Weather"),
        ("synthetic_balanced", "synth-balanced"), ("synthetic_homo", "synth-homo"),
        ("synthetic_hetero", "synth-hetero"),
    ):
        if key in name:
            return label
    return "other"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate routing analysis by dataset.")
    parser.add_argument("--summary", default="results/routing_analysis_summary.csv")
    parser.add_argument("--out", default="results/routing_analysis_by_dataset.csv")
    args = parser.parse_args(argv)

    rows = []
    with open(args.summary, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[dataset_key(row.get("run", ""))].append(row)

    out_rows = []
    fields = ["spread", "margin", "stability", "gain_gate", "gain_val_best", "gain_oracle", "gate", "best_single", "val_best", "oracle"]
    for key, items in sorted(groups.items()):
        out = {"dataset": key, "runs": len(items)}
        for field in fields:
            vals = []
            for item in items:
                try:
                    vals.append(float(item[field]))
                except (TypeError, ValueError, KeyError):
                    pass
            if vals:
                out[f"{field}_mean"] = st.fmean(vals)
                out[f"{field}_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
            else:
                out[f"{field}_mean"] = ""
                out[f"{field}_std"] = ""
        out_rows.append(out)

    out_fields = ["dataset", "runs"] + [f"{f}_{s}" for f in fields for s in ("mean", "std")]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in out_fields})

    def f(x, d=3):
        try:
            return f"{float(x):.{d}f}"
        except (TypeError, ValueError):
            return "-"

    print(f"{'dataset':16s} {'n':>2s} {'spread':>14s} {'gain_gate':>14s} {'gain_val':>14s} {'gain_oracle':>14s}")
    for row in out_rows:
        print(f"{row['dataset']:16s} {row['runs']:>2d} "
              f"{f(row['spread_mean'])}±{f(row['spread_std']):>6s} "
              f"{f(row['gain_gate_mean']):>7s} "
              f"{f(row['gain_val_best_mean']):>7s} "
              f"{f(row['gain_oracle_mean']):>7s}")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
