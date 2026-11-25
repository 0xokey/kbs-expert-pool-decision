"""Collect routing_analysis_*.json files into one table."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect routing analysis tables.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/routing_analysis_summary.csv")
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(Path(args.results_dir).glob("*/routing_analysis_test.json")):
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        rows.append({
            "run": p.get("run"),
            "gate": p.get("gate"),
            "best_single": p.get("best_single", {}).get("rmse"),
            "best_single_expert": p.get("best_single", {}).get("expert"),
            "val_best": p.get("val_best_routing"),
            "oracle": p.get("oracle_routing"),
            "spread": p.get("spread"),
            "stability": p.get("preference_stability"),
            "margin": p.get("selection_margin"),
            "rank_stability": p.get("rank_stability"),
            "gain_gate": p.get("gain_gate"),
            "gain_val_best": p.get("gain_val_best"),
            "gain_oracle": p.get("gain_oracle"),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "gate", "best_single", "best_single_expert", "val_best", "oracle",
              "spread", "margin", "stability", "rank_stability", "gain_gate", "gain_val_best", "gain_oracle"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'run':60s} {'spread':>7s} {'stab':>5s} {'gate':>7s} {'vbest':>7s} {'oracle':>7s} {'gain_v':>7s}")
    for r in rows:
        def f(x, d=4):
            try:
                return f"{float(x):.{d}f}"
            except (TypeError, ValueError):
                return "-"
        print(f"{str(r['run'])[:60]:60s} {f(r['spread'],3):>7s} {f(r['stability'],2):>5s} "
              f"{f(r['gate']):>7s} {f(r['val_best']):>7s} {f(r['oracle']):>7s} {f(r['gain_val_best'],3):>7s}")
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
