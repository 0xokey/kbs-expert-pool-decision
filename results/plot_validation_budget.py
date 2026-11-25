import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=300)
delta = np.logspace(np.log10(0.004), np.log10(0.25), 200)
z = 1.96
colors = {"0.3": "#0072B2", "0.6": "#E69F00", "1.0": "#009E73", "2.0": "#CC79A7"}
for nu, col in colors.items():
    n = (z * float(nu) / delta) ** 2
    ax.plot(delta * 100, n, color=col, lw=1.5, label=rf"$\nu={nu}$")

# aligned validation windows of the eight benchmarks span 14-116
ax.axhspan(14, 116, color="#DDDDDD", alpha=0.55, lw=0)
for n_win in (116, 14):
    ax.axhline(n_win, color="#888888", lw=0.8, ls="--")
bb = dict(fc="white", ec="none", alpha=0.8, pad=1.0)
ax.text(0.45, 132, "ETTm1/ETTm2 (116)", fontsize=7.0, color="#555555", bbox=bb)
ax.text(0.45, 15.6, "Traffic (14)", fontsize=7.0, color="#555555", bbox=bb)
ax.text(24, 40, "aligned windows 14-116", fontsize=7.4, color="#444444", ha="right", bbox=bb)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"relative risk gap $\delta$ (%)")
ax.set_ylabel(r"validation windows needed $n_{\mathrm{req}}$")
ax.set_xlim(0.4, 25); ax.set_ylim(10, 1e6)
ax.grid(True, which="both", alpha=0.22, lw=0.4)
ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.6)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/validation_budget.{ext}", bbox_inches="tight")
print("saved")
