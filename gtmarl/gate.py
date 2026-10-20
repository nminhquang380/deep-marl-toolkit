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
        key = r["method"] + ("_w0" if r["run"].endswith("_w0") else "")
        g[key].append({k: float(v) for k, v in r.items() if k in
                       ("success", "detour", "J", "welfare", "viol", "jain_util", "jain_load")})
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
    checks = [
        ("C1 success >= greedy - 0.02", P and GR and P["success"] >= GR["success"] - 0.02),
        ("C2 J < myopic J", P and MY and P["J"] < MY["J"]),
        ("C3 viol <= myopic viol", P and MY and P["viol"] <= MY["viol"]),
        ("C4 gippo J < ippo J (GNN helps)", P and "ippo" in m and P["J"] < m["ippo"]["J"]),
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
