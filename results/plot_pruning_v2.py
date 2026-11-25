import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
ds = ["ETTh1","ETTh2","ETTm1","ETTm2","Weather","Electricity","Solar","Traffic"]
mix = [0.7085,0.6456,0.7084,0.4863,0.4751,0.4901,0.6043,0.8541]
runs = {
    "ETTh1":[0.6509,0.6507,0.6518], "ETTh2":[0.6073,0.6029,0.6051],
    "ETTm1":[0.6655,0.6701,0.6710], "ETTm2":[0.4604,0.4598,0.4621],
    "Weather":[0.4506,0.4511,0.4517], "Electricity":[0.4541,0.4608,0.4599],
    "Solar":[0.5885,0.6012,0.6070], "Traffic":[0.8388,0.8296,0.8427],
}
single = [0.6511,0.6051,0.6689,0.4608,0.4511,0.4583,0.5989,0.8370]  # Table 3
std = [float(np.std(runs[d], ddof=1)) for d in ds]
gain = [8.1, 6.3, 5.6, 5.2, 5.0, 6.5, 0.9, 2.0]  # Table 3 column

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios":[1.45,1]})
x = np.arange(len(ds)); w = 0.38
ax[0].bar(x-w/2, mix, w, label="best expert in mixture", color="#56B4E9", lw=0)
ax[0].bar(x+w/2, single, w, yerr=std, capsize=2, label="single expert (isolated)",
          color="#0072B2", lw=0, error_kw=dict(lw=0.7, capthick=0.7))
ax[0].set_xticks(x); ax[0].set_xticklabels(ds, rotation=30, ha="right", fontsize=7)
ax[0].set_ylabel("test RMSE (standardised)")
ax[0].set_ylim(0, 0.95)
ax[0].legend(fontsize=7, frameon=False, loc="upper left")
ax[0].grid(axis="y", alpha=0.25, lw=0.4)
ax[0].tick_params(axis="y", labelsize=7)

ax[1].barh(x, gain, color="#009E73", height=0.62, lw=0)
ax[1].axvline(0, color="#333333", lw=0.6)
ax[1].axvline(float(np.mean(gain)), color="#333333", lw=0.6, ls="--")
ax[1].set_yticks(x); ax[1].set_yticklabels(ds, fontsize=7)
ax[1].invert_yaxis()
ax[1].set_xlabel("improvement of the single expert (%)")
ax[1].set_xlim(0, 9.6)
ax[1].grid(axis="x", alpha=0.25, lw=0.4)
ax[1].tick_params(axis="x", labelsize=7)
for i, g in enumerate(gain):
    ax[1].text(g + 0.15, i, f"{g:.1f}", va="center", fontsize=7)

fig.tight_layout(pad=0.3)
fig.savefig("figures/pruning.png", dpi=300)
fig.savefig("figures/pruning.pdf")
print("saved")
