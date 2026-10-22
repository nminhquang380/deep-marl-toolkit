#!/usr/bin/env bash
# Overnight run 2 (~6 h on ~19 cores). Default: N = 40, R_min = 6 Mb/s, loose deadline eta = 3,
# w_nav = 0.5 (failure cost graded by the remaining distance, i.e. shaping without reset).
#   Stage 1 (parallel training)
#     main    : gippo x3, ippo_mlp x3 at eta = 3                -> V-C, CDF of T_i/T_i^min
#     etachk  : gippo trained at eta = 1.2, 1.5, 2.0            -> does MSR(eta') match the CDF?
#     dcomm8  : gippo x2, ippo_mlp x2 at d_comm = 8             -> graph vs k-NN with ~5 neighbours
#     pen     : gippo with indicator / shortfall penalties      -> penalty design preview
#     mappo   : scratch N = 10 and 40, warm start (pilot gippo) -> cooperative benchmark
#   Stage 2: evaluation (SP / MBR at every setting), BR gaps, summary, CDF figure
# Usage: bash gtmarl/scripts/night2.sh [OUT_DIR] [WARM_START_CKPT]
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""   # CPU only: never touch the shared GPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_night2}
WS=${2:-$DATA/gtmarl_pilot/main/gippo_N30_M4_s0/model_last.pt}
IT_MAIN=${IT_MAIN:-1000}; IT_AUX=${IT_AUX:-800}; IT_MAPPO=${IT_MAPPO:-500}
EV_EP=${EV_EP:-1024}; BR_VAL=${BR_VAL:-1024}; BR_TEST=${BR_TEST:-4096}; BR_IT=${BR_IT:-200}
XT=${XT:-}; BRX=${BRX:-}
mkdir -p "$OUT/logs" "$OUT/results"
T="nice -n 10 python -m gtmarl.train --train threads=1 device=cpu $XT"
L="$OUT/logs"
echo "[$(date)] stage 1 start" | tee -a "$L/night.log"

for s in 0 1 2; do
  $T --iters $IT_MAIN --algo gippo    --seed $s --out $OUT/main > $L/main_gippo_s$s.log 2>&1 &
  $T --iters $IT_MAIN --algo ippo_mlp --seed $s --out $OUT/main > $L/main_ippo_mlp_s$s.log 2>&1 &
done
for e in 1.2 1.5 2.0; do
  $T --iters $IT_AUX --algo gippo --seed 0 --out $OUT/etachk --tag eta$e --env eta=$e > $L/eta$e.log 2>&1 &
done
for s in 0 1; do
  $T --iters $IT_AUX --algo gippo    --seed $s --out $OUT/dcomm8 --tag dc8 --env d_comm=8 > $L/dc8_gippo_s$s.log 2>&1 &
  $T --iters $IT_AUX --algo ippo_mlp --seed $s --out $OUT/dcomm8 --tag dc8 --env d_comm=8 > $L/dc8_ippo_mlp_s$s.log 2>&1 &
done
# penalty weights matched to the mean marginal penalty under overload (mu * E[g] ~ 4.5 * 2.4 ~ 11)
$T --iters $IT_AUX --algo gippo --seed 0 --out $OUT/pen --tag indicator --env penalty=indicator mu=11  > $L/pen_ind.log 2>&1 &
$T --iters $IT_AUX --algo gippo --seed 0 --out $OUT/pen --tag shortfall --env penalty=shortfall mu=100 > $L/pen_sf.log 2>&1 &
$T --iters $IT_MAPPO --algo mappo --seed 0 --out $OUT/mappo > $L/mappo_N40.log 2>&1 &
$T --iters $IT_AUX   --algo mappo --seed 0 --out $OUT/mappo --env N=10 > $L/mappo_N10.log 2>&1 &
if [ -f "$WS" ]; then
  $T --iters $IT_MAPPO --algo mappo --seed 0 --out $OUT/mappo --tag ws --train threads=1 device=cpu init_from=$WS $XT \
     > $L/mappo_ws_N40.log 2>&1 &
else
  echo "warm-start checkpoint $WS not found: skipping mappo_ws" | tee -a "$L/night.log"
fi
wait
echo "[$(date)] stage 1 done" | tee -a "$L/night.log"
grep -l -E "Traceback|Error" $L/*.log | sed 's/^/FAILED: /' | tee -a "$L/night.log"

R=$OUT/results
EV="nice -n 10 python -m gtmarl.evaluate --episodes $EV_EP"
$EV --runs $OUT/main/* --heuristics greedy myopic --out $R/main.csv > $L/ev_main.log 2>&1 &
for e in 1.2 1.5 2.0; do
  $EV --runs $OUT/etachk/*eta$e --heuristics greedy myopic --env eta=$e --out $R/eta$e.csv > $L/ev_eta$e.log 2>&1 &
done
$EV --runs $OUT/dcomm8/* --heuristics greedy myopic --env d_comm=8 --out $R/dcomm8.csv > $L/ev_dc8.log 2>&1 &
$EV --runs $OUT/pen/* $OUT/main/gippo_N40_M4_s0 --out $R/pen.csv > $L/ev_pen.log 2>&1 &
$EV --runs $OUT/mappo/* --heuristics greedy myopic --N 10 40 --out $R/mappo.csv > $L/ev_mappo.log 2>&1 &
BR="nice -n 10 python -m gtmarl.br_gap --val_episodes $BR_VAL --eval_episodes $BR_TEST --iters $BR_IT --n_snap 4 $BRX"
br() {  # run, seed, name
  if [ -f "$1/model_last.pt" ]; then $BR --run "$1" --seed "$2" --out "$R/br_$3.json" > "$L/br_$3.log" 2>&1 &
  else echo "skip BR gap for $1 (no checkpoint)" | tee -a "$L/night.log"; fi
}
for s in 0 1 2; do br $OUT/main/gippo_N40_M4_s$s $s gippo_s$s; done
br $OUT/main/ippo_mlp_N40_M4_s0 0 ippo_mlp_s0
br $OUT/mappo/mappo_N40_M4_s0_ws 0 mappo_ws_s0
wait
echo "[$(date)] stage 2 done" | tee -a "$L/night.log"
python -m gtmarl.summarize --res $R | tee $OUT/summary.txt
[ -f $R/main.csv ] && python -m gtmarl.gate --res $R --proposed gippo | tee -a $OUT/summary.txt
python -m gtmarl.plots --runs $OUT >> $OUT/summary.txt 2>&1
echo "figures: $OUT/figs (fig_cdf.pdf)" | tee -a $OUT/summary.txt
