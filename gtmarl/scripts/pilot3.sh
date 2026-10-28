#!/usr/bin/env bash
# Pilot 3: TVT model (separation cost instead of the collision mask, finished UAVs exit and release
# their bearers), proposed egat_cc, before the final sweep.
# Default setting from EnvConfig (N = 50, eta = 1.5, d_comm = 8, d_safe = 2, w_sep = 50) plus SETTING.
#   Stage 0  unit tests; SP / MBR difficulty scan over N x R_min (heuristics only, minutes)
#   Stage 1  training, seeds 0 and 1 unless noted
#     egat_cc (proposed), egat_cc_aux, ippo_mlp, ippo
#     mappo_egat: warm start from the night-3 egat_cc checkpoints + critic warm-up
#     egat_cc with w_sep = 100 (seed 0)               -> separation weight sensitivity
#     egat_cc trained over N in {30, 40, 50, 60} (seed 0)   -> swarm-size randomisation
#   Stage 2  evaluation incl. zero-shot transfer N in {30, 40, 50, 60, 70}, BR gaps, summary, gate
# Usage: SETTING="N=50 R_min=6e6" bash gtmarl/scripts/pilot3.sh [OUT_DIR] [NIGHT3_DIR]
# Knobs: IT (default 800), TH_G (threads per EGAT job, default 2), SETTING (EnvConfig overrides)
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_pilot3}
N3=${2:-$DATA/gtmarl_night3}
IT=${IT:-800}; TH_G=${TH_G:-2}; SETTING=${SETTING:-}
EV_EP=${EV_EP:-1024}; BR_VAL=${BR_VAL:-1024}; BR_TEST=${BR_TEST:-4096}; BR_IT=${BR_IT:-200}
NS=${NS:-"30 40 50 60 70"}
L="$OUT/logs"; R="$OUT/results"; P="$R/parts"
mkdir -p "$L" "$P"
log() { echo "[$(date)] $*" | tee -a "$L/night.log"; }
T="nice -n 10 python -m gtmarl.train"
ENV="--env $SETTING"; [ -z "$SETTING" ] && ENV=""
PN=$(python -c "from gtmarl.config import EnvConfig, apply_overrides; print(apply_overrides(EnvConfig(), '$SETTING'.split()).N)")
N3N=40   # swarm size of the night-3 checkpoints used to warm-start MAPPO

python -m pytest -q gtmarl/tests > "$L/tests.log" 2>&1 || { log "unit tests FAILED, see $L/tests.log"; exit 1; }

log "stage 0: difficulty scan (SP, MBR) with SETTING='$SETTING'"
for r in 6e6 8e6; do
  nice -n 10 python -m gtmarl.evaluate --episodes 128 --heuristics greedy myopic --N 40 45 50 55 60 \
       --env $SETTING R_min=$r --out $P/scan_R$r.csv > $L/scan_R$r.log 2>&1 &
done
wait

log "stage 1 start (N=$PN, IT=$IT, EGAT threads=$TH_G)"
for s in 0 1; do
  for a in ippo_mlp ippo; do
    $T --train threads=1 --iters $IT --algo $a --seed $s --out $OUT/main $ENV > $L/main_${a}_s$s.log 2>&1 &
  done
  for a in egat_cc egat_cc_aux; do
    $T --train threads=$TH_G --iters $IT --algo $a --seed $s --out $OUT/main $ENV > $L/main_${a}_s$s.log 2>&1 &
  done
  WS=$N3/main/egat_cc_N${N3N}_M4_s$s/model_last.pt
  if [ -f "$WS" ]; then
    $T --train threads=$TH_G init_from=$WS critic_warmup=100 lr=1e-4 --iters $IT --algo mappo_egat \
       --seed $s --out $OUT/main $ENV > $L/main_mappo_egat_s$s.log 2>&1 &
  else
    log "warm-start checkpoint $WS not found: mappo_egat s$s from scratch"
    $T --train threads=$TH_G --iters $IT --algo mappo_egat --seed $s --out $OUT/main $ENV \
       > $L/main_mappo_egat_s$s.log 2>&1 &
  fi
done
$T --train threads=$TH_G --iters $IT --algo egat_cc --seed 0 --tag wsep100 --out $OUT/wsep \
   --env $SETTING w_sep=100 > $L/wsep100.log 2>&1 &
$T --train threads=$TH_G "N_set=30,40,50,60" --iters $IT --algo egat_cc --seed 0 --tag nrand --out $OUT/nrand \
   $ENV > $L/nrand.log 2>&1 &
wait
log "stage 1 done"
grep -l -E "Traceback|Error" $L/main_*.log $L/wsep100.log $L/nrand.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"

log "stage 2: evaluation and BR gaps"
EV="nice -n 10 python -m gtmarl.evaluate --episodes $EV_EP"
for a in egat_cc egat_cc_aux ippo_mlp ippo mappo_egat; do
  $EV --runs "$OUT/main/${a}_N${PN}_*" --N $NS $ENV --out $P/main_$a.csv > $L/ev_main_$a.log 2>&1 &
done
for h in greedy myopic; do
  $EV --heuristics $h --N $NS $ENV --out $P/main_$h.csv > $L/ev_main_$h.log 2>&1 &
done
$EV --runs "$OUT/wsep/*" --heuristics greedy myopic --env $SETTING w_sep=100 --out $R/wsep100.csv > $L/ev_wsep.log 2>&1 &
$EV --runs "$OUT/nrand/*" --N $NS $ENV --out $R/nrand.csv > $L/ev_nrand.log 2>&1 &
BR="nice -n 10 python -m gtmarl.br_gap --val_episodes $BR_VAL --eval_episodes $BR_TEST --iters $BR_IT --n_snap 4"
br() {  # run, seed, name
  if [ -f "$1/model_last.pt" ]; then $BR --threads $TH_G --run "$1" --seed "$2" --out "$R/br_$3.json" > "$L/br_$3.log" 2>&1 &
  else log "skip BR gap for $1 (no checkpoint)"; fi
}
for s in 0 1; do
  br $OUT/main/egat_cc_N${PN}_M4_s$s $s egat_cc_s$s
  br $OUT/main/mappo_egat_N${PN}_M4_s$s $s mappo_egat_s$s
done
wait
python - "$P" "$R/main.csv" <<'PY'
import csv, glob, sys
rows = [r for f in sorted(glob.glob(sys.argv[1] + "/main_*.csv")) for r in csv.DictReader(open(f))]
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
cp $P/scan_*.csv $R/ 2>/dev/null
grep -h "FAILED" $L/ev_*.log 2>/dev/null | sort -u | tee -a "$L/night.log"
log "stage 2 done"
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat_cc --N $PN | tee -a $OUT/summary.txt
