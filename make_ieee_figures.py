"""IEEE-column-sized vector figures (PDF) for the IEEEtran version of the article."""
import os, pickle, __main__
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qusin_scenario import PLACES, place_xy
from sagin_v2 import V2, METHODS
__main__.V2 = V2
OUT = os.environ.get("IEEE_FIG_DIR", "../1_Report/Figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "FreeSerif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .4,
    "axes.linewidth": .6, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
v2 = pickle.load(open("results_qusin_v2/results.pkl", "rb")); raw = pickle.load(open("results_qusin_v2/paired_raw.pkl", "rb"))
v1 = pickle.load(open("results_qusin/results.pkl", "rb"))
COL = {"Q-Learning": "#d62728", "Random": "#9e9e9e", "MaxPower+Greedy": "#1f77b4", "BestUniform+Greedy": "#2ca02c",
       "GreedyLB": "#9467bd", "GreedyLB+LEO": "#8c564b", "GreedyLB+LEO(minP)": "#17becf", "SingleHAPS": "#ff7f0e"}
SH = ["Q-Learn", "Random", "MaxPow", "BestUni", "GreedyLB", "LB+LEO", "LB+LEO minP", "SingleHAPS"]
cent = place_xy()

def amap(ax, lay, key, nh, sub=None, title=""):
    col = plt.cm.Set1(np.arange(nh))
    for u, (x, y) in enumerate(lay["gu"]):
        h = lay[key][u]; ax.plot([x, lay["haps"][h][0]], [y, lay["haps"][h][1]], color=col[h], alpha=.3, lw=.5)
    ax.scatter(*lay["gu"].T, c=[col[h] for h in lay[key]], s=9, edgecolor="k", lw=.25, zorder=3)
    for h in range(nh):
        ax.scatter(*lay["haps"][h], marker="^" if sub is None or sub[h] == 0 else "s", s=55, color=col[h], edgecolor="k", lw=.6, zorder=4)
    for (n, *_), (cx, cy) in zip(PLACES, cent):
        ax.annotate(n, (cx, cy), xytext=(0, -9), textcoords="offset points", ha="center", fontsize=5.5, fontweight="bold", color="#333")
    ax.set_title(title + f"  users/HAPS: {np.bincount(lay[key], minlength=nh).tolist()}", fontsize=7)
    ax.set_xlabel("East (km)"); ax.set_aspect("equal")

# Fig A: first model red flag
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.5), sharex=True, sharey=True)
amap(ax[0], v1["layout"], "g_assign", 4, None, "(a) Greedy"); amap(ax[1], v1["layout"], "q_assign", 4, None, "(b) Q-learning")
ax[0].set_ylabel("North (km)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_v1_collapse.pdf"); plt.close(fig)

# Fig D: v2 associations
L = v2["layout"]; fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5), sharex=True, sharey=True)
for a, k, t in zip(ax, ["g_assign", "lb_assign", "q_assign"], ["(a) Greedy", "(b) GreedyLB", "(c) Q-learning"]):
    amap(a, L, k, 4, L["subband"], t)
ax[0].set_ylabel("North (km)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_assoc_v2.pdf"); plt.close(fig)

# Fig B: beam pattern
P = v2["p"]; fig, ax = plt.subplots(figsize=(3.4, 2.1)); off = np.linspace(0, 8, 400)
for g, c in [(20, "#888"), (27, "#1f77b4"), (35, "#d62728")]:
    t3 = np.sqrt(30000 / 10 ** (g / 10)); th = np.degrees(np.arctan(off / 20))
    ax.plot(off, -np.minimum(12 * (th / t3) ** 2, P.sla_db), color=c, lw=1.4, label=f"{g} dBi ($\\theta_{{3dB}}$={t3:.1f}$^\\circ$)")
ax.set_xlabel("Ground distance from beam centre (km), HAPS at 20 km"); ax.set_ylabel("Relative gain (dB)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_beam.pdf"); plt.close(fig)

# Fig C: main comparison
spec = [("outage", "Outage (%)"), ("p5_rate", "5th-pct rate (Mbit/s)"), ("power_w", "Tx power (W)")]
fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.5))
for r, n in enumerate([30, 45]):
    for c, (k, yl) in enumerate(spec):
        ax = axes[r, c]
        for i, m in enumerate(METHODS):
            a = np.array(raw[n][m][k]); ax.bar(i, a.mean(), yerr=a.std() / np.sqrt(len(a)), color=COL[m], capsize=1.5, edgecolor="k", lw=.3, error_kw=dict(lw=.6))
        ax.set_xticks(range(8)); ax.set_xticklabels(SH if r == 1 else [""] * 8, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel(yl); ax.set_title(f"{n} users", fontsize=7.5); ax.grid(axis="x", visible=False); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.pdf"); plt.close(fig)

# Fig E: learning curve
cfg = v2["cfg"]; w = cfg.episodes // 200
sm = np.array([np.convolve(c, np.ones(w) / w, "valid") for c in v2["curves"]]); mu, sd = sm.mean(0), sm.std(0); x = np.arange(mu.size) + w
fig, ax = plt.subplots(figsize=(3.4, 2.1)); ax.plot(x, mu, color=COL["Q-Learning"], lw=1.2); ax.fill_between(x, mu - sd, mu + sd, color=COL["Q-Learning"], alpha=.2)
ax.axvline(cfg.eps_decay_frac * cfg.episodes, color="k", ls="--", lw=.7); ax.text(cfg.eps_decay_frac * cfg.episodes, ax.get_ylim()[0] + .3, " exploration ends", fontsize=6.5)
ax.set_xlabel("Training episode"); ax.set_ylabel("Training reward (sum-rate)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_learning.pdf"); plt.close(fig)

# Fig F: sweeps
S = v2["sweeps"]; names = [("n_users", "Number of users"), ("n_sub", "Frequency sub-bands"), ("beam_gain_dbi", "HAPS beam gain (dBi)")]
STY = {"Q-Learning": ("o", "-"), "GreedyLB": ("s", "--"), "GreedyLB+LEO": ("D", "-"), "GreedyLB+LEO(minP)": ("v", "-"), "BestUniform+Greedy": ("^", ":"), "SingleHAPS": ("x", "-")}
fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.7), sharex="col")
for j, (nm, xl) in enumerate(names):
    vals = list(S[nm].keys())
    for m in ["Q-Learning", "GreedyLB", "GreedyLB+LEO", "GreedyLB+LEO(minP)", "BestUniform+Greedy"] + (["SingleHAPS"] if nm == "n_users" else []):
        for r, k in enumerate(["outage", "power_w"]):
            a = [np.array(S[nm][v][m][k]) for v in vals]
            axes[r, j].errorbar(vals, [q.mean() for q in a], yerr=[q.std() / np.sqrt(len(q)) for q in a], marker=STY[m][0], ls=STY[m][1],
                                capsize=1.5, color=COL[m], lw=1, ms=3.5, label=m, elinewidth=.6)
    axes[1, j].set_xlabel(xl); axes[1, j].set_xticks(vals); axes[0, j].set_ylabel("Outage (%)"); axes[1, j].set_ylabel("Tx power (W)")
    axes[0, j].set_ylim(bottom=0); axes[1, j].set_ylim(bottom=0)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/fig_sweeps.pdf"); plt.close(fig)
print(sorted(os.listdir(OUT)))
