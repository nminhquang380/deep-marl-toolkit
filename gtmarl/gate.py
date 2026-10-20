"""Go / no-go table after the pilot (Phase 0).

python -m gtmarl.gate --res /data/$USER/gtmarl_pilot/results
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
    a = p.parse_args(argv)
    rows = list(csv.DictReader(open(os.path.join(a.res, "main.csv"))))
    g = defaultdict(list)
    for r in rows:
        key = r["method"] + (f"_{r['tag']}" if r.get("tag") else "")
        g[key].append({k: float(v) for k, v in r.items() if k in
                       ("success", "detour", "p_out", "jain_load", "social_cost", "J", "viol")})
    m = {k: {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} for k, v in g.items()}

    print(f"{'method':10s} " + " ".join(f"{k:>9s}" for k in next(iter(m.values()))))
    for k, v in m.items():
        print(f"{k:10s} " + " ".join(f"{x:9.4f}" for x in v.values()))

    br = {}
    for f in glob.glob(os.path.join(a.res, "br_*.json")):
        d = json.load(open(f))
        br[d["algo"]] = d["gap_rel"]
    print("\nrelative BR gap:", {k: f"{100 * v:.1f}%" for k, v in br.items()})

    P, MY, GR = m.get("gippo"), m.get("myopic"), m.get("greedy")
    best = lambda k: min(MY[k], GR[k]) if MY and GR else float("nan")
    checks = [
        ("C1 MSR >= SP MSR - 0.02", P and GR and P["success"] >= GR["success"] - 0.02),
        ("C2 J < min(J_SP, J_MBR)", P and MY and GR and P["J"] < best("J")),
        ("C3 P_out <= min(P_out SP, MBR)", P and MY and GR and P["p_out"] <= best("p_out")),
        ("C4 J < J of IPPO-MLP (graph helps)", P and "ippo_mlp" in m and P["J"] < m["ippo_mlp"]["J"]),
        ("C5 BR gap(gippo) < 5%", "gippo" in br and br["gippo"] < 0.05),
        ("C6 BR gap(mappo) > BR gap(gippo)", "gippo" in br and "mappo" in br and br["mappo"] > br["gippo"]),
        ("C7 graded cost (w=0.1) raises success over w=0", P and "gippo_w0" in m and P["success"] > m["gippo_w0"]["success"]),
    ]
    print()
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    core = all(bool(ok) for _, ok in checks[:3])
    print("\nDECISION:", "GO -> full sweep" if core else "NO-GO -> Phase 1 tuning (C1-C3 must pass)")


if __name__ == "__main__":
    main()
