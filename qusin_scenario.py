"""Emergency backup connectivity for Qusin and nearby villages (Nablus) over a SAGIN.

Scenario: the terrestrial network is down (power cut / damaged towers). Users in
Qusin and nearby villages are served by HAPS platforms that get their
backhaul from LEO satellites. We reuse the repo's channel model, environment and
Q-learning agents unchanged and only add:

  1. a *clustered* user layout around real place locations (not uniform random),
  2. an end-to-end rate per user = min(access rate, backhaul share of its HAPS),
  3. user-level metrics: mean rate, 5th-percentile rate, outage %, Jain fairness,
     and total transmit power.

Coordinates and populations come from Wikipedia village pages (PCBS 2017 preliminary
census). User density is proportional to population; spread sigma is an assumption.

Usage:
    python qusin_scenario.py            # full run  (~1-2 min)
    python qusin_scenario.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace

import numpy as np

from channel import (
    Topology,
    dbm_to_w,
    noise_power_w,
    path_gain_db,
    sample_gain,
)
from config import Config
from env import sum_rate
from trainer import train_one_topology

OUT = os.path.join(os.path.dirname(__file__), "results_qusin")

# (name, lat, lon, population 2017 (PCBS), spread sigma km)
# Coordinates and populations: Wikipedia village pages (PCBS 2017 preliminary census).
PLACES = [
    ("Qusin", 32.23861, 35.18333, 2251, 0.6),
    ("Deir Sharaf", 32.25417, 35.18806, 2949, 0.6),
    ("Beit Iba", 32.23611, 35.20972, 4079, 0.7),
    ("Sarra", 32.21000, 35.19083, 3384, 0.6),
    ("Zawata", 32.24583, 35.22611, 2537, 0.6),
    ("Beit Wazan", 32.23139, 35.21583, 1312, 0.5),
]
LAT0, LON0 = 32.23861, 35.18333  # origin = Qusin centre


def latlon_to_km(lat, lon):
    x = (lon - LON0) * 111.32 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110.57
    return x, y


def place_xy():
    return np.array([latlon_to_km(p[1], p[2]) for p in PLACES])  # [n_place, 2]


def make_clustered_topology(cfg: Config, rng, n_haps_layout=None):
    """Users clustered around the places; HAPS spread over the region; LEOs high."""
    centres = place_xy()
    weights = np.array([p[3] for p in PLACES])
    sig = np.array([p[4] for p in PLACES])
    which = rng.choice(len(PLACES), size=cfg.n_gu, p=weights / weights.sum())
    gu_xy = centres[which] + rng.normal(size=(cfg.n_gu, 2)) * sig[which, None]
    gu = np.vstack([gu_xy.T, np.zeros((1, cfg.n_gu))])

    # HAPS: roughly evenly spread over the region's bounding box (+ station-keeping jitter)
    lo, hi = centres.min(0) - 1.0, centres.max(0) + 1.0
    side = int(np.ceil(np.sqrt(cfg.n_haps)))
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], side + 2)[1:-1],
        np.linspace(lo[1], hi[1], side + 2)[1:-1],
    )
    grid = np.stack([gx.ravel(), gy.ravel()], 1)[: cfg.n_haps]
    haps_xy = grid + rng.normal(size=grid.shape) * 0.5
    haps = np.vstack([haps_xy.T, np.full((1, cfg.n_haps), cfg.height_haps_km)])

    # LEO: somewhere within +-150 km horizontally, at orbit height
    leo_xy = rng.uniform(-150, 150, size=(2, cfg.n_leo))
    leo = np.vstack([leo_xy, np.full((1, cfg.n_leo), cfg.height_leo_km)])

    topo = Topology(
        pg_leo_haps_db=path_gain_db(leo, haps, cfg.freq_hz, cfg.antenna_gain_leo_haps_db),
        pg_haps_gu_db=path_gain_db(haps, gu, cfg.freq_hz, cfg.antenna_gain_haps_gu_db),
    )
    return topo, dict(gu=gu_xy, haps=haps_xy, leo=leo_xy.T, which=which)


# --------------------------------------------------------------------- metrics
def e2e_user_rates(p_leo, p_haps, a_ha, a_ga, g1, g2, noise_w, n_haps):
    """End-to-end per-user rate = min(access rate, its HAPS's backhaul share)."""
    _, rates1 = sum_rate(p_leo, a_ha, g1, noise_w)  # per-HAPS backhaul rate
    _, rates2 = sum_rate(p_haps, a_ga, g2, noise_w)  # per-GU access rate
    served = np.bincount(a_ga, minlength=n_haps)
    backhaul_share = rates1[a_ga] / served[a_ga]
    return np.minimum(rates2, backhaul_share)


def jain(x):
    return float(x.sum() ** 2 / (len(x) * (x**2).sum() + 1e-12))


def evaluate(cfg, topo, policy_fn, rng, n_samples, thr):
    """policy_fn(rng) -> (p_leo_dbm, p_haps_dbm, a_ha, a_ga). Returns metric dict."""
    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)
    all_rates, tot_power, act_h_hist = [], [], []
    for _ in range(n_samples):
        p_l, p_h, a_ha, a_ga = policy_fn(rng)
        g1 = sample_gain(topo.pg_leo_haps_db, rng)
        g2 = sample_gain(topo.pg_haps_gu_db, rng)
        r = e2e_user_rates(p_l, p_h, a_ha, a_ga, g1, g2, noise_w, cfg.n_haps)
        all_rates.append(r)
        act_h = np.bincount(a_ga, minlength=cfg.n_haps) > 0
        act_h_hist.append(act_h.sum())
        act_l = np.bincount(a_ha, minlength=cfg.n_leo) > 0
        tot_power.append(
            (dbm_to_w(p_h) * act_h).sum() + (dbm_to_w(p_l) * act_l).sum()
        )
    R = np.array(all_rates)  # [samples, users]
    mean_user = R.mean(0)  # ergodic rate per user
    return dict(
        sum_rate=float(R.sum(1).mean()),
        p5_rate=float(np.percentile(mean_user, 5)),
        outage=float((mean_user < thr).mean() * 100),
        active_haps=float(np.mean(act_h_hist)),
        jain=jain(mean_user),
        power_w=float(np.mean(tot_power)),
    )


# --------------------------------------------------------------------- policies
def greedy_assoc(topo):
    return topo.pg_leo_haps_db.argmax(0), topo.pg_haps_gu_db.argmax(0)


def make_policies(cfg, topo, qpolicy, rng):
    powers = cfg.power_levels_dbm
    a_ha_g, a_ga_g = greedy_assoc(topo)

    s_ha, s_lp, s_ga, s_hp = qpolicy.state
    q = (
        powers[qpolicy.leo_power.greedy(s_lp)],
        powers[qpolicy.haps_power.greedy(s_hp)],
        qpolicy.haps_assoc.greedy(s_ha),
        qpolicy.gu_assoc.greedy(s_ga),
    )

    def rand(r):
        return (
            powers[r.integers(0, cfg.n_power_levels, cfg.n_leo)],
            powers[r.integers(0, cfg.n_power_levels, cfg.n_haps)],
            r.integers(0, cfg.n_leo, cfg.n_haps),
            r.integers(0, cfg.n_haps, cfg.n_gu),
        )

    def maxp(_):
        return (np.full(cfg.n_leo, cfg.power_max_dbm),
                np.full(cfg.n_haps, cfg.power_max_dbm), a_ha_g, a_ga_g)

    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)

    def best_common_power(a_ha, a_ga):
        best_p, best_v = powers[-1], -1
        for p in powers:
            v = 0.0
            sweep_rng = np.random.default_rng(123)
            for _ in range(40):
                g1 = sample_gain(topo.pg_leo_haps_db, sweep_rng)
                g2 = sample_gain(topo.pg_haps_gu_db, sweep_rng)
                r = e2e_user_rates(np.full(cfg.n_leo, p), np.full(cfg.n_haps, p),
                                   a_ha, a_ga, g1, g2, noise_w, cfg.n_haps)
                v += r.sum()
            if v > best_v:
                best_v, best_p = v, p
        return best_p

    best_p = best_common_power(a_ha_g, a_ga_g)

    def uni(_):
        return (np.full(cfg.n_leo, best_p), np.full(cfg.n_haps, best_p), a_ha_g, a_ga_g)

    # SingleHAPS: ALL users on the one HAPS with the best mean gain (no inter-HAPS
    # interference, one big TDMA cell). Tests whether Q-learning merely rediscovers this.
    h_star = int((10 ** (topo.pg_haps_gu_db / 10)).mean(1).argmax())
    a_ga_s = np.full(cfg.n_gu, h_star)
    p_s = best_common_power(a_ha_g, a_ga_s)

    def single(_):
        return (np.full(cfg.n_leo, p_s), np.full(cfg.n_haps, p_s), a_ha_g, a_ga_s)

    return {
        "Q-Learning": lambda _r: q,
        "Random": rand,
        "MaxPower+Greedy": maxp,
        "BestUniform+Greedy": uni,
        "SingleHAPS": single,
    }


METHODS = ["Q-Learning", "Random", "MaxPower+Greedy", "BestUniform+Greedy", "SingleHAPS"]
METRICS = ["sum_rate", "p5_rate", "outage", "jain", "power_w", "active_haps"]


def run_setting(cfg, n_topo, n_eval, thr, seed):
    master = np.random.default_rng(seed)
    acc = {m: {k: [] for k in METRICS} for m in METHODS}
    curves, last_layout = [], None
    for t in range(n_topo):
        rng = np.random.default_rng(master.integers(2**31))
        topo, layout = make_clustered_topology(cfg, rng)
        curve, qpol = train_one_topology(cfg, topo, rng)
        curves.append(curve)
        pols = make_policies(cfg, topo, qpol, rng)
        for m in METHODS:
            res = evaluate(cfg, topo, pols[m], rng, n_eval, thr)
            for k in METRICS:
                acc[m][k].append(res[k])
        if t == 0:
            last_layout = layout
            layout["q_assign"] = qpol.gu_assoc.greedy(qpol.state[2])
            layout["g_assign"] = greedy_assoc(topo)[1]
    return acc, np.array(curves), last_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    base = Config()
    # Emergency scenario: 3 LEO, 4 HAPS, 30 representative users (each = a group of real users)
    cfg = replace(base, n_leo=3, n_haps=4, n_gu=30, episodes=20000)
    n_topo, n_eval = 10, 100
    if a.quick:
        cfg = replace(cfg, episodes=3000)
        n_topo, n_eval = 3, 30
    THR = 0.05  # outage threshold, bits/s/Hz  (0.05 * 10 MHz = 0.5 Mbit/s per user)

    print(f"Main scenario: {cfg.n_leo} LEO / {cfg.n_haps} HAPS / {cfg.n_gu} users, "
          f"{cfg.episodes} episodes, {n_topo} topologies")
    acc, curves, layout = run_setting(cfg, n_topo, n_eval, THR, a.seed)

    lines = [f"{'method':<20}{'sum-rate':>10}{'p5 rate':>10}{'outage%':>10}{'Jain':>8}{'Tx power W':>12}{'active HAPS':>13}"]
    for m in METHODS:
        v = {k: np.mean(acc[m][k]) for k in METRICS}
        lines.append(f"{m:<20}{v['sum_rate']:>10.2f}{v['p5_rate']:>10.3f}"
                     f"{v['outage']:>10.1f}{v['jain']:>8.2f}{v['power_w']:>12.1f}{v['active_haps']:>13.2f}")
    summary = "\n".join(lines)
    print(summary)
    open(os.path.join(OUT, "summary.txt"), "w").write(summary + "\n")

    # ---- sweep: number of HAPS (planning question: how many do we need?) ----
    sweep = {}
    haps_list = [2, 3, 4, 6]
    for nh in haps_list:
        c = replace(cfg, n_haps=nh, episodes=cfg.episodes if not a.quick else 3000)
        r, _, _ = run_setting(c, n_topo, n_eval // 2, THR, a.seed + 1)
        sweep[nh] = r
        print(f"  n_haps={nh}: " + "  ".join(
            f"{m}: outage={np.mean(r[m]['outage']):.1f}%" for m in ("Q-Learning", "BestUniform+Greedy")))

    # ---- sweep: number of users (load) ----
    sweep_users = {}
    users_list = [15, 30, 45, 60]
    for nu in users_list:
        c = replace(cfg, n_gu=nu, episodes=cfg.episodes if not a.quick else 3000)
        r, _, _ = run_setting(c, n_topo, n_eval // 2, THR, a.seed + 2)
        sweep_users[nu] = r
        print(f"  n_users={nu}: " + "  ".join(
            f"{m}: outage={np.mean(r[m]['outage']):.1f}%" for m in ("Q-Learning", "BestUniform+Greedy")))

    # ---- diagnostic: is the backhaul or the access tier the bottleneck? ----
    diag = []
    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)
    drng = np.random.default_rng(a.seed + 3)
    for nh in haps_list:
        c = replace(cfg, n_haps=nh)
        frac = []
        for _ in range(10):
            tp, _ = make_clustered_topology(c, drng)
            a_ha, a_ga = greedy_assoc(tp)
            for _ in range(20):
                g1 = sample_gain(tp.pg_leo_haps_db, drng)
                g2 = sample_gain(tp.pg_haps_gu_db, drng)
                _, r1 = sum_rate(np.full(c.n_leo, c.power_max_dbm), a_ha, g1, noise_w)
                _, r2 = sum_rate(np.full(c.n_haps, c.power_max_dbm), a_ga, g2, noise_w)
                served = np.bincount(a_ga, minlength=nh)
                frac.append((r1[a_ga] / served[a_ga] < r2).mean())
        diag.append(f"n_haps={nh}: users limited by backhaul = {100 * np.mean(frac):.0f}%")
    print("\n".join(diag))
    open(os.path.join(OUT, "diagnostic.txt"), "w").write("\n".join(diag) + "\n")

    np.savez(os.path.join(OUT, "raw.npz"), curves=curves,
             **{f"{m}__{k}": np.array(acc[m][k]) for m in METHODS for k in METRICS})
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_haps", "method"] + METRICS)
        for nh, r in sweep.items():
            for m in METHODS:
                w.writerow([nh, m] + [f"{np.mean(r[m][k]):.4f}" for k in METRICS])

    import pickle
    pickle.dump(dict(acc=acc, curves=curves, layout=layout, sweep=sweep, sweep_users=sweep_users, users_list=users_list, cfg=cfg,
                     haps_list=haps_list, thr=THR),
                open(os.path.join(OUT, "results.pkl"), "wb"))
    print("saved to", OUT)


if __name__ == "__main__":
    main()
