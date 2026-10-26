#!/usr/bin/env bash
# Overnight run 3: graph encoder diagnostics with beacon observations (obs_version 2), and MAPPO fix.
# Default setting: N = 40, R_min = 6 Mb/s, eta = 1.5, d_comm = 8 (200 m), w_fail = w_nav = 0.5.
#   Stage 0  timing of one iteration (gippo vs egat), printed to the log
#   Stage 1  training (seeds 0, 1)
#     A1  ippo (no neighbours), ippo_mlp (k-NN), gippo (mean), egat (proposed),
#         egat_aux (+ congestion head), egat_cc (+ agent-specific centralised critic, own reward)
#     A3  CTDE-MAPPO fix: warm start from the night-2 gippo checkpoints in their own setting
#         (obs_version 1, eta = 3, d_comm = 4), critic warm-up, lr 1e-4
#   Stage 2  evaluation incl. zero-shot transfer N in {20, 40, 60, 80} (A4), BR gaps, summary, gate
# Usage: bash gtmarl/scripts/night3.sh [OUT_DIR] [NIGHT2_DIR]
# Knobs: IT (iterations, default 800), TH_G (threads per EGAT job, default 2)
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""   # CPU only: never touch the shared GPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_night3}
N2=${2:-$DATA/gtmarl_night2}
IT=${IT:-800}; IT_MAPPO=${IT_MAPPO:-800}; TH_G=${TH_G:-2}
EV_EP=${EV_EP:-1024}; BR_VAL=${BR_VAL:-1024}; BR_TEST=${BR_TEST:-4096}; BR_IT=${BR_IT:-200}
mkdir -p "$OUT/logs" "$OUT/results"
L="$OUT/logs"; R="$OUT/results"
T="nice -n 10 python -m gtmarl.train --train device=cpu"
log() { echo "[$(date)] $*" | tee -a "$L/night.log"; }

python -m pytest -q gtmarl/tests > "$L/tests.log" 2>&1 || { log "unit tests FAILED, see $L/tests.log"; exit 1; }

log "stage 0: timing (2 iterations each)"
for a in gippo egat; do
  th=1; [ "${a#egat}" != "$a" ] && th=$TH_G
  $T threads=$th eval_every=100000 save_every=100000 --algo $a --iters 2 --out $OUT/timing > $L/timing_$a.log 2>&1
  python - "$OUT/timing/${a}_N40_M4_s0/progress.csv" "$a" "$IT" <<'PY' | tee -a "$L/night.log"
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); t = float(r[-1]["time_min"]) * 60 / len(r)
print(f"  {sys.argv[2]:6s}: {t:6.1f} s/iter (early iterations are the slowest) -> {int(sys.argv[3])} iters ~ {t * int(sys.argv[3]) / 3600:.1f} h upper bound")
PY
done

log "stage 1 start (IT=$IT, EGAT threads=$TH_G)"
for s in 0 1; do
  for a in ippo ippo_mlp gippo; do
    $T threads=1 --iters $IT --algo $a --seed $s --out $OUT/main > $L/main_${a}_s$s.log 2>&1 &
  done
  for a in egat egat_aux egat_cc; do
    $T threads=$TH_G --iters $IT --algo $a --seed $s --out $OUT/main > $L/main_${a}_s$s.log 2>&1 &
  done
  WS=$N2/main/gippo_N40_M4_s$s/model_last.pt
  if [ -f "$WS" ]; then
    $T threads=1 init_from=$WS critic_warmup=100 lr=1e-4 --iters $IT_MAPPO --algo mappo --seed $s \
       --tag ws2 --out $OUT/mappo --env obs_version=1 eta=3 d_comm=4 > $L/mappo_ws2_s$s.log 2>&1 &
  else
    log "warm-start checkpoint $WS not found: skipping mappo_ws2 s$s"
  fi
done
wait
log "stage 1 done"
grep -l -E "Traceback|Error" $L/main_*.log $L/mappo_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"

EV="nice -n 10 python -m gtmarl.evaluate --episodes $EV_EP"
$EV --runs "$OUT/mappo/*" "$N2/main/gippo_N40_M4_s0" "$N2/main/gippo_N40_M4_s1" --heuristics greedy myopic \
    --env obs_version=1 eta=3 d_comm=4 --out $R/mappo.csv > $L/ev_mappo.log 2>&1 &
BR="nice -n 10 python -m gtmarl.br_gap --val_episodes $BR_VAL --eval_episodes $BR_TEST --iters $BR_IT --n_snap 4"
br() {  # run, seed, name, threads
  if [ -f "$1/model_last.pt" ]; then $BR --threads "$4" --run "$1" --seed "$2" --out "$R/br_$3.json" > "$L/br_$3.log" 2>&1 &
  else log "skip BR gap for $1 (no checkpoint)"; fi
}
for s in 0 1; do
  br $OUT/main/egat_N40_M4_s$s $s egat_s$s $TH_G
  br $OUT/mappo/mappo_N40_M4_s${s}_ws2 $s mappo_ws2_s$s 1
done
br $OUT/main/egat_cc_N40_M4_s0 0 egat_cc_s0 $TH_G
br $OUT/main/ippo_mlp_N40_M4_s0 0 ippo_mlp_s0 1
wait
log "stage 2 done"
bash "$(dirname "${BASH_SOURCE[0]}")/night3_post.sh" "$OUT" "$N2"   # main + transfer evaluation, summary, gate
