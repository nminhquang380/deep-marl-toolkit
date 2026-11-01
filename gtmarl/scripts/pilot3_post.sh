#!/usr/bin/env bash
# Finish pilot 3 without retraining or re-evaluating: merge main.csv, print summary and gate now,
# then compute the BR gaps with a lighter configuration and print the final summary.
# Run it detached so a closed terminal does not kill it:
#   nohup bash gtmarl/scripts/pilot3_post.sh > gtmarl/data/gtmarl_pilot3/post.out 2>&1 &
# Knobs: BR_ENVS (32), BR_IT (100), BR_VAL (512), BR_TEST (2048), BR_TH (threads per job, 4)
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_pilot3}
L="$OUT/logs"; R="$OUT/results"; P="$R/parts"
BR_ENVS=${BR_ENVS:-32}; BR_IT=${BR_IT:-100}; BR_VAL=${BR_VAL:-512}; BR_TEST=${BR_TEST:-2048}; BR_TH=${BR_TH:-4}
PN=$(python -c "from gtmarl.config import EnvConfig; print(EnvConfig().N)")
log() { echo "[$(date)] $*" | tee -a "$L/night.log"; }

python - "$P" "$R/main.csv" <<'PY'
import csv, glob, sys
rows = [r for f in sorted(glob.glob(sys.argv[1] + "/main_*.csv")) for r in csv.DictReader(open(f))]
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} rows -> {sys.argv[2]}")
PY
cp $P/scan_*.csv $R/ 2>/dev/null
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat_cc --N $PN | tee -a $OUT/summary.txt

log "post: BR gaps (envs=$BR_ENVS, iters=$BR_IT, val=$BR_VAL, test=$BR_TEST, threads=$BR_TH)"
BR="nice -n 10 python -m gtmarl.br_gap --n_envs $BR_ENVS --val_episodes $BR_VAL --eval_episodes $BR_TEST --iters $BR_IT --n_snap 2"
for job in "egat_cc_N${PN}_M4_s0 0 egat_cc_s0" "egat_cc_N${PN}_M4_s1 1 egat_cc_s1" "ippo_mlp_N${PN}_M4_s0 0 ippo_mlp_s0"; do
  set -- $job
  [ -f "$OUT/main/$1/model_last.pt" ] && \
    $BR --threads $BR_TH --run "$OUT/main/$1" --seed $2 --out "$R/br_$3.json" > "$L/br_$3.log" 2>&1 &
done
wait
log "post: BR gaps done"
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat_cc --N $PN | tee -a $OUT/summary.txt
