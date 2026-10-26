#!/usr/bin/env bash
# Re-run stage 2 of night3 (evaluation incl. zero-shot transfer, summary, gate) on the trained runs.
# The BR-gap files of night3 are kept. One evaluation process per method, merged into main.csv.
# Usage: bash gtmarl/scripts/night3_post.sh [OUT_DIR] [NIGHT2_DIR]
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_night3}
N2=${2:-$DATA/gtmarl_night2}
EV_EP=${EV_EP:-1024}; NS=${NS:-"20 40 60 80"}
L="$OUT/logs"; R="$OUT/results"; P="$R/parts"
mkdir -p "$L" "$P"
EV="nice -n 10 python -m gtmarl.evaluate --episodes $EV_EP"
echo "[$(date)] post: evaluation start" | tee -a "$L/night.log"
for a in ippo ippo_mlp gippo egat egat_aux egat_cc; do
  $EV --runs "$OUT/main/${a}_N40_*" --N $NS --out $P/main_$a.csv > $L/ev_main_$a.log 2>&1 &
done
for h in greedy myopic; do
  $EV --heuristics $h --N $NS --out $P/main_$h.csv > $L/ev_main_$h.log 2>&1 &
done
if [ -d "$N2/dcomm8" ]; then
  $EV --runs "$N2/dcomm8/*" --heuristics greedy myopic --env eta=3 --N $NS \
      --out $R/transfer_night2_dc8.csv > $L/ev_n2.log 2>&1 &
fi
wait
python - "$P" "$R/main.csv" <<'PY'
import csv, glob, sys
rows = [r for f in sorted(glob.glob(sys.argv[1] + "/main_*.csv")) for r in csv.DictReader(open(f))]
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} rows -> {sys.argv[2]}")
PY
grep -h "FAILED\|Error" $L/ev_main_*.log $L/ev_n2.log 2>/dev/null | sort -u | tee -a "$L/night.log"
echo "[$(date)] post: evaluation done" | tee -a "$L/night.log"
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat | tee -a $OUT/summary.txt
python -m gtmarl.plots --runs $OUT --proposed egat >> $OUT/summary.txt 2>&1 || echo "plots failed (see summary.txt)"
