"""Compact summary of every results/*.csv and br_*.json of a run directory.

python -m gtmarl.summarize --res gtmarl/data/gtmarl_night1/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLS = ("success", "detour", "p_out", "viol", "jain_load", "J", "nbr_mean", "nbr_gt_k",
        "F@1.2", "F@1.5", "F@2.0")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True)
    a = p.parse_args(argv)
    for f in sorted(glob.glob(os.path.join(a.res, "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows or "method" not in rows[0]:
            continue
        g = defaultdict(list)
        for r in rows:
            key = (r["method"] + (f"[{r['tag']}]" if r.get("tag") else ""), r["N"], r.get("M", ""))
            g[key].append(r)
        print(f"\n== {os.path.basename(f)}")
        print(f"{'method':22s} {'N':>3s} {'B':>2s} {'n':>2s} " + " ".join(f"{c:>9s}" for c in COLS))
        for (m, N, B), rs in g.items():
            vals = []
            for c in COLS:
                v = np.array([float(r.get(c, "nan") or "nan") for r in rs])
                vals.append(f"{np.nanmean(v):9.4f}")
            print(f"{m:22s} {N:>3s} {B:>2s} {len(rs):>2d} " + " ".join(vals))
    brs = sorted(glob.glob(os.path.join(a.res, "br_*.json")))
    if brs:
        print("\n== best-response gap (paired test; gap > 0 means a profitable unilateral deviation)")
        for f in brs:
            d = json.load(open(f))
            print(f"{os.path.basename(f):34s} J0(pi) {d['J0_policy']:8.3f}  J0(BR) {d['J0_br']:8.3f}  "
                  f"gap {d['gap']:7.3f} +/- {d['gap_se']:.3f}  ({100 * d['gap_rel']:5.1f}%)  "
                  f"MSR0 {d['succ0_policy']:.3f}->{d['succ0_br']:.3f}  best it {d['best_iter']}")


if __name__ == "__main__":
    main()
