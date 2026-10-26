"""Publication-quality figures for the Qusin SAGIN emergency-connectivity study."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qusin_scenario import PLACES, place_xy, METHODS, OUT

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COL = {"Q-Learning": "#d62728", "Random": "#9e9e9e",
       "MaxPower+Greedy": "#1f77b4", "BestUniform+Greedy": "#2ca02c", "SingleHAPS": "#ff7f0e"}
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
D = pickle.load(open(os.path.join(OUT, "results.pkl"), "rb"))
acc, curves, lay, cfg = D["acc"], D["curves"], D["layout"], D["cfg"]

# ---------- Fig 1: topology / association map ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
cent = place_xy()
for ax, key, title in zip(axes, ["g_assign", "q_assign"],
                          ["(a) Greedy: strongest-signal association", "(b) Q-Learning association"]):
    ha_col = plt.cm.Set1(np.arange(cfg.n_haps))
    for u, (x, y) in enumerate(lay["gu"]):
        h = lay[key][u]; hx, hy = lay["haps"][h]
        ax.plot([x, hx], [y, hy], color=ha_col[h], alpha=0.35, lw=0.8, zorder=1)
    ax.scatter(lay["gu"][:, 0], lay["gu"][:, 1], c=[ha_col[h] for h in lay[key]],
               s=22, edgecolor="k", linewidth=0.3, zorder=3, label="Users")
    ax.scatter(lay["haps"][:, 0], lay["haps"][:, 1], marker="^", s=170, c=ha_col,
               edgecolor="k", zorder=4, label="HAPS")
    for (n, *_), (cx, cy) in zip(PLACES, cent):
        ax.annotate(n, (cx, cy), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold", color="#333")
    counts = np.bincount(lay[key], minlength=cfg.n_haps)
    ax.set_title(title + f"\nusers per HAPS: {counts.tolist()}", fontsize=10)
    ax.set_xlabel("East (km)"); ax.set_aspect("equal")
axes[0].set_ylabel("North (km)")
axes[0].legend(loc="upper left", fontsize=8, frameon=True)
fig.suptitle("Qusin area: user clusters, HAPS and association (one random topology)", y=1.02, fontsize=11)
fig.savefig(os.path.join(FIG, "fig1_topology_association.png")); plt.close(fig)

# ---------- Fig 2: main comparison (outage / p5 rate / power) ----------
def ms(m, k): a = np.array(acc[m][k]); return a.mean(), a.std()
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, k, yl, ttl in zip(axes, ["outage", "p5_rate", "power_w", "active_haps"],
        ["Users in outage (%)", "5th-percentile user rate (bit/s/Hz)", "Total transmit power (W)", "Active HAPS (of 4)"],
        ["(a) Outage  (lower is better)", "(b) Worst-case users  (higher is better)", "(c) Energy use  (lower is better)", "(d) HAPS switched on"]):
    for i, m in enumerate(METHODS):
        mu, sd = ms(m, k)
        ax.bar(i, mu, yerr=sd, color=COL[m], capsize=3, edgecolor="k", linewidth=0.5)
        ax.text(i, mu + sd + 0.01 * ax.get_ylim()[1], f"{mu:.1f}" if k != "p5_rate" else f"{mu:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q-Learn", "Random", "MaxPow", "BestUni", "Single\nHAPS"], fontsize=8.5)
    ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10); ax.grid(axis="x", visible=False); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_main_comparison.png")); plt.close(fig)

# ---------- Fig 3: learning curve ----------
win = cfg.episodes // 200
sm = np.array([np.convolve(c, np.ones(win) / win, "valid") for c in curves])
mu, sd = sm.mean(0), sm.std(0); x = np.arange(mu.size) + win
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(x, mu, color=COL["Q-Learning"], lw=1.8, label="Q-Learning (mean of 10 topologies)")
ax.fill_between(x, mu - sd, mu + sd, color=COL["Q-Learning"], alpha=0.2)
ax.axvline(cfg.eps_decay_frac * cfg.episodes, color="k", ls="--", lw=0.8)
ax.text(cfg.eps_decay_frac * cfg.episodes, ax.get_ylim()[0] + 0.05, "  exploration ends", fontsize=8)
ax.set_xlabel("Training episode"); ax.set_ylabel("Training reward (tier-1 + tier-2 sum-rate)")
ax.set_title("Q-learning convergence"); ax.legend(frameon=False)
fig.savefig(os.path.join(FIG, "fig3_learning_curve.png")); plt.close(fig)

# ---------- Fig 4: sweeps ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, sw, lst, xl in [(axes[0], D["sweep_users"], D["users_list"], "Number of users (load)"),
                        (axes[1], D["sweep"], D["haps_list"], "Number of HAPS (full frequency reuse)")]:
    for m in ["Q-Learning", "MaxPower+Greedy", "BestUniform+Greedy", "SingleHAPS"]:
        mu = [np.mean(sw[v][m]["outage"]) for v in lst]; sd = [np.std(sw[v][m]["outage"]) for v in lst]
        ax.errorbar(lst, mu, yerr=sd, marker="o", capsize=3, color=COL[m], label=m, lw=1.6)
    ax.set_xticks(lst); ax.set_xlabel(xl); ax.set_ylabel("Users in outage (%)"); ax.legend(frameon=False, fontsize=8)
axes[0].set_title("(a) Outage vs. load  (4 HAPS, 3 LEO)", fontsize=10)
axes[1].set_title("(b) Outage vs. HAPS count  (30 users, 3 LEO)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_sweeps.png")); plt.close(fig)
print(os.listdir(FIG))
