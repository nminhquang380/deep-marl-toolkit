#!/usr/bin/env bash
# Pilot 4: TVT model with the corrected separation rule (||q_i - q_j|| <= 1 voxel, w_sep = 30),
# N = 50, eta = 1.5, d_comm = 8. Run it inside tmux:  bash gtmarl/scripts/pilot4.sh
#   Stage 0  unit tests; SP / MBR scan over N (heuristics only, minutes)
#   Stage 1  training, seeds 0 and 1: egat_cc (proposed), ippo_mlp, ippo (no neighbours),
#            mappo_egat (warm start from the pilot-3 egat_cc checkpoints, critic warm-up 200, lr 5e-5)
#   Stage 2  evaluation incl. transfer N in {30, 40, 50, 60}, light BR gaps, learning curves, summary, gate
# Knobs: IT (default 1000), TH_G (threads per EGAT job, 2), TX (TrainConfig overrides for every learner,
#        default "n_envs=16 epochs=4": ~2x faster per iteration), BR_* (see below), SETTING (EnvConfig overrides)
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_pilot4}
P3=${2:-$DATA/gtmarl_pilot3}
IT=${IT:-1000}; TH_G=${TH_G:-2}; SETTING=${SETTING:-}; TX=${TX:-"n_envs=16 epochs=4"}
EV_EP=${EV_EP:-1024}; NS=${NS:-"30 40 50 60"}
BR_ENVS=${BR_ENVS:-32}; BR_IT=${BR_IT:-100}; BR_VAL=${BR_VAL:-512}; BR_TEST=${BR_TEST:-2048}; BR_TH=${BR_TH:-4}
L="$OUT/logs"; R="$OUT/results"; P="$R/parts"
mkdir -p "$L" "$P"
log() { echo "[$(date)] $*" | tee -a "$L/night.log"; }
if pgrep -f "gtmarl.train .*--out $OUT/" > /dev/null; then
  echo "another run is already writing to $OUT; stop it first"; exit 1
fi
T="nice -n 10 python -m gtmarl.train"
ENV="--env $SETTING"; [ -z "$SETTING" ] && ENV=""
PN=$(python -c "from gtmarl.config import EnvConfig, apply_overrides; print(apply_overrides(EnvConfig(), '$SETTING'.split()).N)")

python -m pytest -q gtmarl/tests > "$L/tests.log" 2>&1 || { log "unit tests FAILED, see $L/tests.log"; exit 1; }
log "stage 0: SP / MBR scan"
nice -n 10 python -m gtmarl.evaluate --episodes 128 --heuristics greedy myopic --N 40 45 50 55 $ENV \
     --out $R/scan.csv > $L/scan.log 2>&1

log "stage 1 start (N=$PN, IT=$IT, EGAT threads=$TH_G, TX=$TX)"
for s in 0 1; do
  for a in ippo_mlp ippo; do
    $T --train threads=1 $TX --iters $IT --algo $a --seed $s --out $OUT/main $ENV > $L/main_${a}_s$s.log 2>&1 &
  done
  $T --train threads=$TH_G $TX --iters $IT --algo egat_cc --seed $s --out $OUT/main $ENV > $L/main_egat_cc_s$s.log 2>&1 &
  WS=$P3/main/egat_cc_N${PN}_M4_s$s/model_last.pt
  if [ -f "$WS" ]; then
    $T --train threads=$TH_G $TX init_from=$WS critic_warmup=200 lr=5e-5 --iters $IT --algo mappo_egat --seed $s \
       --out $OUT/main $ENV > $L/main_mappo_egat_s$s.log 2>&1 &
  else
    log "warm-start checkpoint $WS not found: mappo_egat s$s from scratch"
    $T --train threads=$TH_G $TX --iters $IT --algo mappo_egat --seed $s --out $OUT/main $ENV \
       > $L/main_mappo_egat_s$s.log 2>&1 &
  fi
done
wait
log "stage 1 done"
grep -l -E "Traceback|Error" $L/main_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"

log "stage 2: evaluation"
EV="nice -n 10 python -m gtmarl.evaluate --episodes $EV_EP"
for a in egat_cc ippo_mlp ippo mappo_egat; do
  $EV --runs "$OUT/main/${a}_N${PN}_*" --N $NS $ENV --out $P/main_$a.csv > $L/ev_main_$a.log 2>&1 &
done
for h in greedy myopic; do
  $EV --heuristics $h --N $NS $ENV --out $P/main_$h.csv > $L/ev_main_$h.log 2>&1 &
done
wait
python - "$P" "$R/main.csv" <<'PY'
import csv, glob, sys
rows = [r for f in sorted(glob.glob(sys.argv[1] + "/main_*.csv")) for r in csv.DictReader(open(f))]
fields = list(dict.fromkeys(k for r in rows for k in r))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
python - "$OUT/main" <<'PY' | tee $OUT/curves.txt
# learning curves (evaluation every 50 iterations): is training converged?
import csv, glob, os, sys
for d in sorted(glob.glob(sys.argv[1] + "/*")):
    f = os.path.join(d, "progress.csv")
    if not os.path.exists(f):
        continue
    rows = [r for r in csv.DictReader(open(f)) if r.get("eval/success")]
    pts = rows[3::4] or rows
    print(f"{os.path.basename(d):28s} " + "  ".join(
        f"it{int(r['iter']) + 1}: MSR {float(r['eval/success']):.2f} J {float(r['eval/J']):.0f}" for r in pts)
        + f"  ({float(rows[-1]['time_min']) / max(1, int(rows[-1]['iter']) + 1):.2f} min/it)")
PY
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat_cc --N $PN | tee -a $OUT/summary.txt

log "stage 2: BR gaps (envs=$BR_ENVS, iters=$BR_IT, val=$BR_VAL, test=$BR_TEST, threads=$BR_TH)"
BR="nice -n 10 python -m gtmarl.br_gap --n_envs $BR_ENVS --val_episodes $BR_VAL --eval_episodes $BR_TEST --iters $BR_IT --n_snap 2"
for job in "egat_cc_N${PN}_M4_s0 0 egat_cc_s0" "egat_cc_N${PN}_M4_s1 1 egat_cc_s1" \
           "mappo_egat_N${PN}_M4_s0 0 mappo_egat_s0" "ippo_mlp_N${PN}_M4_s0 0 ippo_mlp_s0"; do
  set -- $job
  [ -f "$OUT/main/$1/model_last.pt" ] && \
    $BR --threads $BR_TH --run "$OUT/main/$1" --seed $2 --out "$R/br_$3.json" > "$L/br_$3.log" 2>&1 &
done
wait
log "stage 2 done"
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
python -m gtmarl.gate --res $R --proposed egat_cc --N $PN | tee -a $OUT/summary.txt
cat $OUT/curves.txt >> $OUT/summary.txt
