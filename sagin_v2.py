"""Model v2: emergency connectivity around Qusin (Nablus) with a more realistic HAPS access tier.

What changes relative to the repo's environment (repo files are NOT modified):

  1. Spot beams.  Each HAPS points a beam at the user it is serving. The desired link
     gets the peak gain; interference from another HAPS reaches a user through that
     HAPS's beams, attenuated by an antenna pattern that depends on the angle between
     the beam direction and the victim user (3GPP-style parabolic main lobe, capped at
     a side-lobe level).  Beams are scheduled round-robin, so the interference a user
     sees from HAPS m is the *average* over m's served users.
  2. Frequency reuse.  The band is split into ``n_sub`` sub-bands; neighbouring HAPS
     use different sub-bands (co-channel interference only inside a sub-band, but each
     HAPS only owns B / n_sub of spectrum).
  3. Capacity.  A HAPS serves at most ``kmax`` users; extra users are *blocked*
     (rate 0).  The Q-learning reward therefore penalises overloading a HAPS.

4.  Directional backhaul.  Each HAPS points a 40 dBi dish at its serving LEO, so
     interference from the other LEOs is attenuated by the same kind of pattern,
     evaluated at the actual angular separation of the LEOs seen from that HAPS.

Users' end-to-end rate is min(access rate, share of the HAPS backhaul).

Usage:   python sagin_v2.py            # full run
         python sagin_v2.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, replace

import numpy as np

from agents import AgentGroup, epsilon_at
from channel import dbm_to_w, noise_power_w, path_gain_db, sample_gain
from config import Config
from qusin_scenario import PLACES, make_clustered_topology, place_xy  # noqa: F401
from trainer import Policy

OUT = os.path.join(os.path.dirname(__file__), "results_qusin_v2")


@dataclass
class V2:
    g_max_dbi: float = 35.0  # HAPS spot-beam peak gain
    sla_db: float = 25.0  # side-lobe attenuation cap
    n_sub: int = 2  # frequency sub-bands
    kmax: int = 12  # max users per HAPS
    thr_mbps: float = 1.0  # outage threshold (per-user average rate)


@dataclass
class Topo2:
    pg_leo_haps_db: np.ndarray  # [n_leo, n_haps]
    pg_haps_gu_db: np.ndarray  # [n_haps, n_gu]  (peak beam gain, free-space)
    rel: np.ndarray  # [n_haps, n_gu, n_gu] pattern attenuation: beam on i seen by j (linear)
    subband: np.ndarray  # [n_haps]
    rel1: np.ndarray = None  # [n_haps, n_leo, n_leo] LEO-dish pattern attenuation (linear)


# ------------------------------------------------------------------ topology
def make_topology_v2(cfg, p: V2, rng):
    topo0, lay = make_clustered_topology(cfg, rng)  # uses cfg.antenna_gain_haps_gu_db
    n_h, n_u = cfg.n_haps, cfg.n_gu
    H = np.column_stack([lay["haps"], np.full(n_h, cfg.height_haps_km)])  # [n_h,3]
    U = np.column_stack([lay["gu"], np.zeros(n_u)])  # [n_u,3]
    D = U[None, :, :] - H[:, None, :]
    V = D / np.linalg.norm(D, axis=2, keepdims=True)
    cos = np.clip(np.einsum("mik,mjk->mij", V, V), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos))
    theta3 = np.sqrt(30000.0 / 10 ** (p.g_max_dbi / 10))  # 3 dB beamwidth (deg)
    att_db = np.minimum(12.0 * (theta / theta3) ** 2, p.sla_db)
    rel = 10 ** (-att_db / 10)
    order = np.argsort(lay["haps"][:, 0])
    sub = np.zeros(n_h, dtype=int)
    sub[order] = np.arange(n_h) % p.n_sub
    L = np.column_stack([lay["leo"], np.full(cfg.n_leo, cfg.height_leo_km)])  # [n_leo,3]
    DL = L[None, :, :] - H[:, None, :]
    VL = DL / np.linalg.norm(DL, axis=2, keepdims=True)
    cos1 = np.clip(np.einsum("hik,hjk->hij", VL, VL), -1.0, 1.0)
    th1 = np.degrees(np.arccos(cos1))
    th3_1 = np.sqrt(30000.0 / 10 ** (cfg.antenna_gain_leo_haps_db / 10))
    rel1 = 10 ** (-np.minimum(12.0 * (th1 / th3_1) ** 2, p.sla_db) / 10)
    topo = Topo2(topo0.pg_leo_haps_db, topo0.pg_haps_gu_db, rel, sub, rel1)
    return topo, lay


# ------------------------------------------------------------------ access tier
def access_rates(p_w, assoc, topo: Topo2, gain, noise_w, p: V2):
    """Per-user access rate (bit/s/Hz of the *total* band), served mask, users/HAPS."""
    n_h, n_u = gain.shape
    adm = np.zeros((n_h, n_u), dtype=bool)
    for m in range(n_h):
        idx = np.flatnonzero(assoc == m)
        if idx.size > p.kmax:  # admission control: keep the strongest users
            idx = idx[np.argsort(-topo.pg_haps_gu_db[m, idx])[: p.kmax]]
        adm[m, idx] = True
    n_adm = adm.sum(1)
    mean_rel = np.einsum("mi,mij->mj", adm.astype(float), topo.rel) / np.maximum(n_adm, 1)[:, None]
    imat = p_w[:, None] * gain * mean_rel * (n_adm > 0)[:, None]  # [m, j]
    cc = topo.subband[:, None] == topo.subband[None, :]
    j = np.arange(n_u)
    inter = (imat * cc[:, assoc]).sum(0) - imat[assoc, j]
    desired = p_w[assoc] * gain[assoc, j]
    sinr = desired / (inter + noise_w / p.n_sub)
    rate = np.log2(1.0 + sinr) / p.n_sub / np.maximum(n_adm[assoc], 1)
    served = adm[assoc, j]
    return rate * served, served, n_adm


def tier1_rates(p_w, a_ha, topo: Topo2, g1, noise_w):
    """LEO -> HAPS backhaul with directional dishes. Returns (sum, per-HAPS rate)."""
    n_leo, n_h = g1.shape
    cnt = np.bincount(a_ha, minlength=n_leo)
    active = cnt > 0
    rx = p_w[:, None] * g1  # [l, h]
    h = np.arange(n_h)
    desired = rx[a_ha, h]
    rel = topo.rel1[h, a_ha, :]  # [h, l']
    inter = ((rx.T * rel) * active[None, :]).sum(1) - desired * topo.rel1[h, a_ha, a_ha]
    rate = np.log2(1.0 + desired / (inter + noise_w)) / cnt[a_ha]
    return float(rate.sum()), rate


def e2e_rates(p_leo_dbm, p_haps_dbm, a_ha, a_ga, topo, g1, g2, noise_w, p):
    _, r1 = tier1_rates(dbm_to_w(p_leo_dbm), a_ha, topo, g1, noise_w)  # per-HAPS backhaul rate
    r2, served, n_adm = access_rates(dbm_to_w(p_haps_dbm), a_ga, topo, g2, noise_w, p)
    share = r1[a_ga] / np.maximum(n_adm[a_ga], 1)
    return np.minimum(r2, share), served, n_adm


# ------------------------------------------------------------------ Q-learning (repo agents)
def train_v2(cfg, p: V2, topo: Topo2, rng):
    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)
    powers = cfg.power_levels_dbm
    ha = AgentGroup(cfg.n_haps, cfg.n_leo, cfg.n_leo)
    lp = AgentGroup(cfg.n_leo, cfg.n_power_levels, cfg.n_power_levels)
    ga = AgentGroup(cfg.n_gu, cfg.n_haps, cfg.n_haps)
    hp = AgentGroup(cfg.n_haps, cfg.n_power_levels, cfg.n_power_levels)
    s_ha = rng.integers(0, cfg.n_leo, cfg.n_haps)
    s_lp = rng.integers(0, cfg.n_power_levels, cfg.n_leo)
    s_ga = rng.integers(0, cfg.n_haps, cfg.n_gu)
    s_hp = rng.integers(0, cfg.n_power_levels, cfg.n_haps)
    curve = np.empty(cfg.episodes)
    for ep in range(cfg.episodes):
        eps = epsilon_at(ep, cfg)
        a_ha = ha.select(s_ha, eps, rng)
        a_lp = lp.select(s_lp, eps, rng)
        a_ga = ga.select(s_ga, eps, rng)
        a_hp = hp.select(s_hp, eps, rng)
        g1 = sample_gain(topo.pg_leo_haps_db, rng)
        g2 = sample_gain(topo.pg_haps_gu_db, rng)
        r1, rates1 = tier1_rates(dbm_to_w(powers[a_lp]), a_ha, topo, g1, noise_w)
        rates2, _, _ = access_rates(dbm_to_w(powers[a_hp]), a_ga, topo, g2, noise_w, p)
        r2 = rates2.sum()
        ha.update(s_ha, a_ha, rates1, a_ha, cfg.alpha, cfg.gamma)
        lp.update(s_lp, a_lp, r1, a_lp, cfg.alpha, cfg.gamma)
        ga.update(s_ga, a_ga, rates2, a_ga, cfg.alpha, cfg.gamma)
        hp.update(s_hp, a_hp, r2, a_hp, cfg.alpha, cfg.gamma)
        s_ha, s_lp, s_ga, s_hp = a_ha, a_lp, a_ga, a_hp
        curve[ep] = r1 + r2
    return curve, Policy(ha, lp, ga, hp, (s_ha, s_lp, s_ga, s_hp))


# ------------------------------------------------------------------ evaluation
def jain(x):
    return float(x.sum() ** 2 / (len(x) * (x**2).sum() + 1e-12))


METHODS = ["Q-Learning", "Random", "MaxPower+Greedy", "BestUniform+Greedy", "GreedyLB", "GreedyLB+LEO", "GreedyLB+LEO(minP)", "SingleHAPS"]
METRICS = ["sum_rate", "p5_rate", "outage", "jain", "power_w", "active_haps", "blocked"]


def evaluate(cfg, p, topo, policy_fn, rng, n_samples):
    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)
    thr = p.thr_mbps / (cfg.bandwidth_hz / 1e6)  # bit/s/Hz
    R, power, act, blk = [], [], [], []
    for _ in range(n_samples):
        pl, ph, a_ha, a_ga = policy_fn(rng)
        g1 = sample_gain(topo.pg_leo_haps_db, rng)
        g2 = sample_gain(topo.pg_haps_gu_db, rng)
        r, served, n_adm = e2e_rates(pl, ph, a_ha, a_ga, topo, g1, g2, noise_w, p)
        R.append(r)
        act_h = n_adm > 0
        act_l = np.bincount(a_ha, minlength=cfg.n_leo) > 0
        power.append((dbm_to_w(ph) * act_h).sum() + (dbm_to_w(pl) * act_l).sum())
        act.append(act_h.sum())
        blk.append(100 * (1 - served.mean()))
    R = np.array(R)
    mu = R.mean(0)
    return dict(
        sum_rate=float(R.sum(1).mean()) * cfg.bandwidth_hz / 1e6,  # Mbit/s
        p5_rate=float(np.percentile(mu, 5)) * cfg.bandwidth_hz / 1e6,  # Mbit/s
        outage=float((mu < thr).mean() * 100),
        jain=jain(mu),
        power_w=float(np.mean(power)),
        active_haps=float(np.mean(act)),
        blocked=float(np.mean(blk)),
    )


# ------------------------------------------------------------------ baselines
def greedy_assoc(topo):
    return topo.pg_leo_haps_db.argmax(0), topo.pg_haps_gu_db.argmax(0)


def lb_assoc(topo, kmax):
    """Capacity-aware greedy: strongest users first, each takes best HAPS with room."""
    n_h, n_u = topo.pg_haps_gu_db.shape
    cnt = np.zeros(n_h, dtype=int)
    a = np.zeros(n_u, dtype=int)
    for j in np.argsort(-topo.pg_haps_gu_db.max(0)):
        for m in np.argsort(-topo.pg_haps_gu_db[:, j]):
            if cnt[m] < kmax:
                a[j] = m
                cnt[m] += 1
                break
        else:
            a[j] = topo.pg_haps_gu_db[:, j].argmax()
    return a


def lb_leo_assoc(topo, n_leo):
    """Balanced backhaul: each HAPS takes its best LEO that still has room (cap = ceil(H/L))."""
    n_h = topo.pg_leo_haps_db.shape[1]
    cap = int(np.ceil(n_h / n_leo))
    cnt = np.zeros(n_leo, dtype=int)
    a = np.zeros(n_h, dtype=int)
    for h in np.argsort(-topo.pg_leo_haps_db.max(0)):
        for l in np.argsort(-topo.pg_leo_haps_db[:, h]):
            if cnt[l] < cap:
                a[h] = l
                cnt[l] += 1
                break
    return a


def make_policies(cfg, p, topo, qpol):
    powers = cfg.power_levels_dbm
    noise_w = noise_power_w(cfg.bandwidth_hz, cfg.noise_figure_db)
    a_ha_g, a_ga_g = greedy_assoc(topo)
    s_ha, s_lp, s_ga, s_hp = qpol.state
    q = (powers[qpol.leo_power.greedy(s_lp)], powers[qpol.haps_power.greedy(s_hp)],
         qpol.haps_assoc.greedy(s_ha), qpol.gu_assoc.greedy(s_ga))

    def best_power(a_ga, a_ha=a_ha_g):
        best, bv = powers[-1], -1.0
        for pw in powers:
            r = np.random.default_rng(123)
            v = 0.0
            for _ in range(30):
                g1 = sample_gain(topo.pg_leo_haps_db, r)
                g2 = sample_gain(topo.pg_haps_gu_db, r)
                e, _, _ = e2e_rates(np.full(cfg.n_leo, pw), np.full(cfg.n_haps, pw),
                                    a_ha, a_ga, topo, g1, g2, noise_w, p)
                v += e.sum()
            if v > bv:
                bv, best = v, pw
        return best

    thr = p.thr_mbps / (cfg.bandwidth_hz / 1e6)

    def min_power(a_ga, a_ha):
        """Smallest common power whose outage on this topology is <= 1% (energy-aware heuristic)."""
        for pw in powers:
            r = np.random.default_rng(321)
            R = []
            for _ in range(30):
                g1 = sample_gain(topo.pg_leo_haps_db, r)
                g2 = sample_gain(topo.pg_haps_gu_db, r)
                e, _, _ = e2e_rates(np.full(cfg.n_leo, pw), np.full(cfg.n_haps, pw),
                                    a_ha, a_ga, topo, g1, g2, noise_w, p)
                R.append(e)
            if (np.mean(R, 0) < thr).mean() <= 0.01:
                return pw
        return powers[-1]

    def fixed(a_ga, pw, a_ha=a_ha_g):
        return lambda _r: (np.full(cfg.n_leo, pw), np.full(cfg.n_haps, pw), a_ha, a_ga)

    def rand(r):
        return (powers[r.integers(0, cfg.n_power_levels, cfg.n_leo)],
                powers[r.integers(0, cfg.n_power_levels, cfg.n_haps)],
                r.integers(0, cfg.n_leo, cfg.n_haps), r.integers(0, cfg.n_haps, cfg.n_gu))

    a_lb = lb_assoc(topo, p.kmax)
    a_ha_lb = lb_leo_assoc(topo, cfg.n_leo)
    h_star = int((10 ** (topo.pg_haps_gu_db / 10)).mean(1).argmax())
    a_single = np.full(cfg.n_gu, h_star)
    return {
        "Q-Learning": lambda _r: q,
        "Random": rand,
        "MaxPower+Greedy": fixed(a_ga_g, powers[-1]),
        "BestUniform+Greedy": fixed(a_ga_g, best_power(a_ga_g)),
        "GreedyLB": fixed(a_lb, best_power(a_lb)),
        "GreedyLB+LEO": fixed(a_lb, best_power(a_lb, a_ha_lb), a_ha_lb),
        "GreedyLB+LEO(minP)": fixed(a_lb, min_power(a_lb, a_ha_lb), a_ha_lb),
        "SingleHAPS": fixed(a_single, best_power(a_single)),
    }


# ------------------------------------------------------------------ runner
def run_setting(cfg, p, n_topo, n_eval, seed):
    master = np.random.default_rng(seed)
    acc = {m: {k: [] for k in METRICS} for m in METHODS}
    curves, layout = [], None
    for t in range(n_topo):
        rng = np.random.default_rng(master.integers(2**31))
        topo, lay = make_topology_v2(cfg, p, rng)
        curve, qpol = train_v2(cfg, p, topo, rng)
        curves.append(curve)
        pols = make_policies(cfg, p, topo, qpol)
        for m in METHODS:
            res = evaluate(cfg, p, topo, pols[m], rng, n_eval)
            for k in METRICS:
                acc[m][k].append(res[k])
        if t == 0:
            layout = dict(lay, q_assign=qpol.gu_assoc.greedy(qpol.state[2]),
                          g_assign=greedy_assoc(topo)[1], lb_assign=lb_assoc(topo, p.kmax),
                          subband=topo.subband)
    return acc, np.array(curves), layout


def table(acc):
    L = [f"{'method':<20}{'sum Mbps':>9}{'p5 Mbps':>9}{'outage%':>9}{'Jain':>7}{'Tx W':>7}{'HAPS on':>9}{'blocked%':>10}"]
    for m in METHODS:
        v = {k: np.mean(acc[m][k]) for k in METRICS}
        L.append(f"{m:<20}{v['sum_rate']:>9.1f}{v['p5_rate']:>9.2f}{v['outage']:>9.1f}{v['jain']:>7.2f}"
                 f"{v['power_w']:>7.1f}{v['active_haps']:>9.2f}{v['blocked']:>10.1f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = V2()
    cfg = replace(Config(), n_leo=3, n_haps=4, n_gu=30, episodes=20000,
                  antenna_gain_haps_gu_db=p.g_max_dbi)
    n_topo, n_eval = 10, 100
    if a.quick:
        cfg, n_topo, n_eval = replace(cfg, episodes=3000), 3, 30

    print(f"MAIN: {cfg.n_leo} LEO / {cfg.n_haps} HAPS / {cfg.n_gu} users, kmax={p.kmax}, "
          f"sub-bands={p.n_sub}, beam={p.g_max_dbi} dBi, {cfg.episodes} episodes, {n_topo} topologies", flush=True)
    acc, curves, layout = run_setting(cfg, p, n_topo, n_eval, a.seed)
    tab = table(acc)
    print(tab, flush=True)
    open(os.path.join(OUT, "summary.txt"), "w").write(tab + "\n")

    sweeps = {}

    def sweep(name, values, make):
        sweeps[name] = {}
        for v in values:
            c, pp = make(v)
            r, _, _ = run_setting(c, pp, n_topo, n_eval // 2, a.seed + 1)
            sweeps[name][v] = r
            print(f"  {name}={v}: " + "  ".join(
                f"{m}: outage={np.mean(r[m]['outage']):.1f}%" for m in ("Q-Learning", "GreedyLB+LEO", "GreedyLB+LEO(minP)")),
                flush=True)

    sweep("n_users", [15, 30, 45, 60], lambda v: (replace(cfg, n_gu=v), p))
    sweep("n_sub", [1, 2, 4], lambda v: (cfg, replace(p, n_sub=v)))
    sweep("beam_gain_dbi", [20, 27, 35],
          lambda v: (replace(cfg, antenna_gain_haps_gu_db=float(v)), replace(p, g_max_dbi=float(v))))

    pickle.dump(dict(acc=acc, curves=curves, layout=layout, sweeps=sweeps, cfg=cfg, p=p),
                open(os.path.join(OUT, "results.pkl"), "wb"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
