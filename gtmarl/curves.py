"""Learning curves of a set of runs (evaluation every eval_every iterations).

python -m gtmarl.curves gtmarl/data/sweep/A/main
"""
import csv
import glob
import os
import sys


def main(argv=None):
    for root in (argv or sys.argv[1:]):
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            f = os.path.join(d, "progress.csv")
            if not os.path.exists(f):
                continue
            rows = [r for r in csv.DictReader(open(f)) if r.get("eval/success")]
            if not rows:
                continue
            pts = rows[3::4] or rows
            print(f"{os.path.basename(d):34s} " + "  ".join(
                f"it{int(r['iter']) + 1}: MSR {float(r['eval/success']):.2f} J {float(r['eval/J']):.0f}"
                for r in pts) + f"  ({float(rows[-1]['time_min']) / max(1, int(rows[-1]['iter']) + 1):.2f} min/it)")


if __name__ == "__main__":
    main()
