"""Go / no-go table after the pilot (Phase 0).

python -m gtmarl.gate --res gtmarl/data/gtmarl_pilot/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True)
    p.add_argument("--proposed", default="gippo", help="egat | gippo | ...")
    p.add_argument("--N", default="40", help="swarm size of the comparison (transfer rows are ignored)")
    a = p.parse_args(argv)
    rows = [r for r in csv.DictReader(open(os.path.join(a.res, "main.csv")))
            if r["N"] == a.N and r.get("N_train", a.N) in ("", "0", a.N)]
    g = defaultdict(list)
    for r in rows:
        key = r["method"] + (f"_{r['tag']}" if r.get("tag") else "")
        g[key].append({k: float(v) for k, v in r.items() if k in
                       ("success", "detour", "p_out", "jain_load", "social_cost", "J", "viol",
                        "nbr_mean", "nbr_gt_k")})
    m = {k: {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} for k, v in g.items()}
    if not m:
        print(f"gate: no rows with N = {a.N} in main.csv"); return

    print(f"{'method':10s} " + " ".join(f"{k:>9s}" for k in next(iter(m.values()))))
    for k, v in m.items():
        print(f"{k:10s} " + " ".join(f"{x:9.4f}" for x in v.values()))

    brs = defaultdict(list)
    for f in glob.glob(os.path.join(a.res, "br_*.json")):
        d = json.load(open(f))
        brs["mappo" if d["algo"].startswith("mappo") else d["algo"]].append(d)
    br = {k: float(np.mean([d["gap_rel"] for d in v])) for k, v in brs.items()}
    print()
    for k, v in brs.items():
        for d in v:
            print(f"BR gap {k:9s}: J0(pi) {d['J0_policy']:.3f} -> J0(BR) {d['J0_br']:.3f}, "
                  f"gap {d['gap']:.3f} +/- {d.get('gap_se', float('nan')):.3f} "
                  f"({100 * d['gap_rel']:.1f}%), best iter {d.get('best_iter', '?')}")

    PR = a.proposed
    P, MY, GR = m.get(PR), m.get("myopic"), m.get("greedy")
    best = lambda k: min(MY[k], GR[k]) if MY and GR else float("nan")
    checks = [
        ("C1 MSR >= SP MSR - 0.02", P and GR and P["success"] >= GR["success"] - 0.02),
        ("C2 J < min(J_SP, J_MBR)", P and MY and GR and P["J"] < best("J")),
        ("C3 P_out <= min(P_out SP, MBR)", P and MY and GR and P["p_out"] <= best("p_out")),
        ("C4 J < J of IPPO-MLP (graph helps)", P and "ippo_mlp" in m and P["J"] < m["ippo_mlp"]["J"]),
        ("C5 BR gap(proposed) < 5%", PR in br and br[PR] < 0.05),
        ("C6 BR gap(mappo) > BR gap(proposed), MAPPO competent", PR in br and "mappo" in br
         and br["mappo"] > br[PR]
         and np.mean([d["succ0_policy"] for d in brs["mappo"]]) > 0.8),
    ]
    print(f"\nproposed = {PR}")
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    core = all(bool(ok) for _, ok in checks[:3])
    print("\nDECISION:", "GO -> full sweep" if core else "NO-GO -> Phase 1 tuning (C1-C3 must pass)")


if __name__ == "__main__":
    main()
