# When Does a Time-Series Forecast Need an Expert Pool?

Code and data for the paper *"How Many Experts Do We Need? A Knowledge-Based
Decision Procedure for Expert Pools in Time-Series Forecasting"* (Knowledge-Based
Systems).

The paper asks a decision question rather than proposing another router: given a
validation split, does the evidence support a channel-specific expert over one
global expert? The rule compares the two leading candidates on paired validation
windows, commits a channel only when the top-two gap clears its sampling noise,
falls back to one validated expert otherwise, and prunes experts that are never
selected. Pruning leaves every prediction unchanged.

## Repository layout

```
train.py                 train a pool (or a single expert) from a JSON config
predict.py               evaluate a trained pool and export per-channel predictions
configs/                 experiment configs (ETT / Weather / Electricity / Solar / Traffic, h=96/192/336)
training/                training loops, evaluation, health checks
network/                 diffusion experts, gate, and the mixture
datasets/generic_csv.py  CSV dataset loader with chronological splits
tools/                   certified routing, budget sweep, bootstraps, exports
results/                 plotting scripts for Figures 2 and 3
tests/                   smoke tests
csv_data/                put the benchmark CSVs here (git-ignored)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x on CPU (the reported runs used a 16 GB
CPU-only machine) and on a single CUDA GPU. Set `"device": "cpu"` in a config if
no GPU is available.

## Data

The eight real benchmarks follow the TSLib/TFB protocol (context 336, horizon 96,
chronological splits, StandardScaler fitted on the training split). Download them
and place the CSVs in `csv_data/` with these names:

| file | source |
| --- | --- |
| `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` | https://github.com/zhouhaoyi/ETDataset |
| `weather.csv`, `electricity.csv`, `traffic.csv`, `solar.csv` | https://github.com/thuml/Time-Series-Library |

The three controlled sets (homogeneous, heterogeneous, balanced) are generated
with `python tools/make_synthetic_dataset.py`.

## Reproducing the main results

Train a ten-expert pool on ETTh1:

```bash
python train.py --config configs/bench_etth1_h96_direct_es.json --seed 42
```

Certify channels and write a routing table:

```bash
python tools/certified_routing.py --run-dir results/<run_name> --tau-mode auto \
    --z 1.96 --fallback best_single --out routes.json
```

Sweep the validation budget:

```bash
python tools/certified_budget_sweep.py --run-dir results/<run_name> --dataset ETTh1
```

Figures 2 and 3 are produced by `results/plot_pruning_v2.py` and
`results/plot_validation_budget.py`; both read the per-window expert losses
exported by `tools/export_expert_losses.py`.

## Reported cost

On ETTh1 with the 16 GB CPU-only machine used in the paper: one isolated expert
trains in 2.1 minutes, the full ten-expert mixture in 9.7 minutes (4.6x), and the
checkpoint shrinks from 9.1 MB to 0.83 MB (11x). Measured serving latency is
0.204 ms versus 0.525 ms per window (2.6x).

## Citation

```bibtex
@article{wang2025experts,
  title   = {How Many Experts Do We Need? A Knowledge-Based Decision Procedure for Expert Pools in Time-Series Forecasting},
  author  = {Wang, Hongbing and Deng, Wenpan},
  journal = {Knowledge-Based Systems},
  year    = {2025}
}
```

## License

MIT. See `LICENSE`.
