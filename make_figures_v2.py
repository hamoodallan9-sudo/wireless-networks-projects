"""Figures for model v2 (spot beams + frequency reuse + HAPS capacity)."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qusin_scenario import PLACES, place_xy
import __main__
from sagin_v2 import METHODS, OUT, V2
__main__.V2 = V2  # results.pkl was written by running sagin_v2.py as a script

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
    "savefig.bbox": "tight"})
COL = {"Q-Learning": "#d62728", "Random": "#9e9e9e", "MaxPower+Greedy": "#1f77b4",
       "BestUniform+Greedy": "#2ca02c", "GreedyLB": "#9467bd", "GreedyLB+LEO": "#8c564b", "GreedyLB+LEO(minP)": "#17becf", "SingleHAPS": "#ff7f0e"}
SHORT = {"Q-Learning": "Q-Learn", "Random": "Random", "MaxPower+Greedy": "MaxPow",
         "BestUniform+Greedy": "BestUni", "GreedyLB": "GreedyLB", "GreedyLB+LEO": "LB+LEO", "GreedyLB+LEO(minP)": "LB+LEO\nminP", "SingleHAPS": "Single\nHAPS"}
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
D = pickle.load(open(os.path.join(OUT, "results.pkl"), "rb"))
acc, curves, lay, cfg, P = D["acc"], D["curves"], D["layout"], D["cfg"], D["p"]

# ---- Fig 1: association maps ----
fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharex=True, sharey=True)
cent = place_xy(); mk = ["^", "s"]
for ax, key, ttl in zip(axes, ["g_assign", "lb_assign", "q_assign"],
        ["(a) Greedy (strongest signal)", "(b) GreedyLB (capacity-aware)", "(c) Q-Learning"]):
    col = plt.cm.Set1(np.arange(cfg.n_haps))
    for u, (x, y) in enumerate(lay["gu"]):
        h = lay[key][u]; ax.plot([x, lay["haps"][h][0]], [y, lay["haps"][h][1]], color=col[h], alpha=.3, lw=.8)
    ax.scatter(*lay["gu"].T, c=[col[h] for h in lay[key]], s=22, edgecolor="k", lw=.3, zorder=3)
    for h in range(cfg.n_haps):
        ax.scatter(*lay["haps"][h], marker=mk[lay["subband"][h]], s=190, color=col[h], edgecolor="k", zorder=4)
    for (n, *_), (cx, cy) in zip(PLACES, cent):
        ax.annotate(n, (cx, cy), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#333")
    cnt = np.bincount(lay[key], minlength=cfg.n_haps)
    ax.set_title(ttl + f"\nusers per HAPS: {cnt.tolist()} (max {P.kmax})", fontsize=10)
    ax.set_xlabel("East (km)"); ax.set_aspect("equal")
axes[0].set_ylabel("North (km)")
fig.suptitle("Qusin area: association under the v2 model (marker shape = frequency sub-band)", y=1.02, fontsize=11)
fig.savefig(os.path.join(FIG, "fig1_topology_association.png")); plt.close(fig)

# ---- Fig 2: main comparison, 30 topologies, 30 and 45 users (rows) ----
RAW = pickle.load(open(os.path.join(OUT, "paired_raw.pkl"), "rb"))
spec = [("outage", "Users in outage (%)", "Outage (<1 Mbit/s), lower=better"),
        ("p5_rate", "5th-pct user rate (Mbit/s)", "Worst-case users, higher=better"),
        ("power_w", "Total transmit power (W)", "Energy use, lower=better"),
        ("blocked", "Blocked users (%)", "Admission blocking, lower=better")]
fig, axes = plt.subplots(4, 2, figsize=(9.5, 11.5))
for r, n in enumerate([30, 45]):
    for ax, (k, yl, t) in zip(axes[:, r], spec):
        for i, m in enumerate(METHODS):
            a_ = np.array(RAW[n][m][k]); mu, se = a_.mean(), a_.std() / np.sqrt(len(a_))
            ax.bar(i, mu, yerr=se, color=COL[m], capsize=3, edgecolor="k", lw=.5)
            ax.text(i, mu + se, f"{mu:.1f}" if k != "p5_rate" else f"{mu:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=8)
        ax.set_ylabel(yl, fontsize=9); ax.set_title(f"{n} users: {t}", fontsize=9.5)
        ax.grid(axis="x", visible=False); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_main_comparison.png")); plt.close(fig)

# ---- Fig 3: learning curve ----
win = cfg.episodes // 200
sm = np.array([np.convolve(c, np.ones(win) / win, "valid") for c in curves]); mu, sd = sm.mean(0), sm.std(0)
x = np.arange(mu.size) + win
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(x, mu, color=COL["Q-Learning"], lw=1.8, label="Q-Learning (mean of 10 topologies)")
ax.fill_between(x, mu - sd, mu + sd, color=COL["Q-Learning"], alpha=.2)
ax.axvline(cfg.eps_decay_frac * cfg.episodes, color="k", ls="--", lw=.8)
ax.text(cfg.eps_decay_frac * cfg.episodes, ax.get_ylim()[1], "  exploration ends", fontsize=9, va="top")
ax.set_xlabel("Training episode"); ax.set_ylabel("Training reward (tier-1 + tier-2 sum-rate)")
ax.set_title("Q-learning convergence (v2 model)"); ax.legend(frameon=False)
fig.savefig(os.path.join(FIG, "fig3_learning_curve.png")); plt.close(fig)

# ---- Fig 4: sweeps (outage and power), error bars = standard error over topologies ----
S = D["sweeps"]
names = [("n_users", "Number of users (load)", "Outage vs load"),
         ("n_sub", "Frequency sub-bands", "Outage vs frequency reuse"),
         ("beam_gain_dbi", "HAPS beam peak gain (dBi)", "Outage vs beam gain")]
STY = {"Q-Learning": ("o", "-"), "GreedyLB": ("s", "--"), "GreedyLB+LEO": ("D", "-"), "GreedyLB+LEO(minP)": ("v", "-"),
       "BestUniform+Greedy": ("^", ":"), "SingleHAPS": ("x", "-")}
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex="col")
for j, (nm, xl, tt) in enumerate(names):
    vals = list(S[nm].keys())
    for m in ["Q-Learning", "GreedyLB+LEO", "GreedyLB+LEO(minP)", "BestUniform+Greedy"] + (["SingleHAPS"] if nm == "n_users" else []):
        for r, k in enumerate(["outage", "power_w"]):
            a = [np.array(S[nm][v][m][k]) for v in vals]
            mu = [x.mean() for x in a]; se = [x.std() / np.sqrt(len(x)) for x in a]
            axes[r, j].errorbar(vals, mu, yerr=se, marker=STY[m][0], ls=STY[m][1], capsize=3,
                                color=COL[m], lw=1.5, ms=6, label=m, alpha=.9)
    axes[0, j].set_title(tt, fontsize=10); axes[1, j].set_xlabel(xl)
    axes[0, j].set_ylabel("Users in outage (%)"); axes[1, j].set_ylabel("Total transmit power (W)")
    axes[0, j].set_ylim(bottom=0); axes[1, j].set_ylim(bottom=0); axes[1, j].set_xticks(vals)
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_sweeps.png")); plt.close(fig)

# ---- Fig 5: beam pattern explainer ----
fig, ax = plt.subplots(figsize=(6.5, 3.6))
off = np.linspace(0, 8, 400)  # km on the ground
for g, c in [(20, "#999"), (27, "#1f77b4"), (35, "#d62728")]:
    th3 = np.sqrt(30000 / 10 ** (g / 10)); th = np.degrees(np.arctan(off / 20))
    ax.plot(off, -np.minimum(12 * (th / th3) ** 2, P.sla_db), color=c, lw=2, label=f"{g} dBi (3 dB beamwidth {th3:.1f}°)")
ax.set_xlabel("Distance from beam centre on the ground (km), HAPS at 20 km")
ax.set_ylabel("Relative antenna gain (dB)"); ax.set_title("HAPS spot-beam pattern used in v2"); ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(FIG, "fig5_beam_pattern.png")); plt.close(fig)
print(os.listdir(FIG))
