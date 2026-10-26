"""Larger runs (30 topologies each, fresh seed) + paired Q-Learning vs baselines with bootstrap CIs."""
import pickle, numpy as np
from dataclasses import replace
from sagin_v2 import *

p = V2()
rng = np.random.default_rng(0)


def ci(d):
    b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


out, raw = [], {}
for n_gu in (30, 45):
    cfg = replace(Config(), n_leo=3, n_haps=4, n_gu=n_gu, episodes=20000, antenna_gain_haps_gu_db=p.g_max_dbi)
    acc, _, _ = run_setting(cfg, p, 30, 100, 777 + n_gu)
    raw[n_gu] = acc
    out += [f"=== {n_gu} users, 30 topologies ===", table(acc), "",
            "Paired difference Q-Learning minus baseline: mean [95% bootstrap CI]",
            "(negative outage / power / blocked = Q-Learning better; positive rate = Q-Learning better)"]
    for base in ["GreedyLB+LEO(minP)", "GreedyLB+LEO", "GreedyLB", "BestUniform+Greedy"]:
        for k in ["outage", "p5_rate", "power_w", "sum_rate"]:
            d = np.array(acc["Q-Learning"][k]) - np.array(acc[base][k])
            m, lo, hi = ci(d)
            out.append(f"  vs {base:<20} {k:<9} {m:8.2f}  [{lo:7.2f}, {hi:7.2f}]")
    out.append("")
    print("\n".join(out[-17:]), flush=True)
open(OUT + "/paired_stats.txt", "w").write("\n".join(out) + "\n")
pickle.dump(raw, open(OUT + "/paired_raw.pkl", "wb"))
