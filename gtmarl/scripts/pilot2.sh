#!/usr/bin/env bash
# Pilot round 2 (~2.5-3 h): fixes and diagnostics after round 1, reusing its runs.
#   gippo_msg (message-MLP graph) x2, ippo_mlp seed 1, ippo (no neighbours), CTDE-MAPPO with
#   agent-specific critic, CTDE-MAPPO warm-started from IPPO-MLP; then the new BR-gap estimator.
# Usage: bash gtmarl/scripts/pilot2.sh [OUT_DIR] [ITERS]
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""   # CPU only: never touch the shared GPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
OUT=${1:-$DATA/gtmarl_pilot}
ITERS=${2:-1500}
mkdir -p "$OUT/logs" "$OUT/results"
rm -rf "$OUT"/main/mappo_N30_M4_s0 "$OUT"/results/br_*.json      # old critic / old estimator
PY="nice -n 10 python -m gtmarl.train --iters $ITERS --train threads=1 device=cpu --out $OUT/main"
$PY --algo gippo_msg --seed 0 > "$OUT/logs/gippo_msg_s0.log" 2>&1 &
$PY --algo gippo_msg --seed 1 > "$OUT/logs/gippo_msg_s1.log" 2>&1 &
$PY --algo ippo_mlp  --seed 1 > "$OUT/logs/ippo_mlp_s1.log" 2>&1 &
$PY --algo ippo      --seed 0 > "$OUT/logs/ippo_s0.log" 2>&1 &
$PY --algo mappo     --seed 0 > "$OUT/logs/mappo_s0.log" 2>&1 &
$PY --algo mappo     --seed 0 --tag ws --train threads=1 init_from="$OUT/main/ippo_mlp_N30_M4_s0/model_last.pt" \
    > "$OUT/logs/mappo_ws_s0.log" 2>&1 &
wait
BR="nice -n 10 python -m gtmarl.br_gap"
for r in gippo_N30_M4_s0 gippo_msg_N30_M4_s0 ippo_mlp_N30_M4_s0 mappo_N30_M4_s0 mappo_N30_M4_s0_ws; do
  $BR --run "$OUT/main/$r" --out "$OUT/results/br_$r.json" > "$OUT/logs/br_$r.log" 2>&1 &
done
wait
python -m gtmarl.evaluate --runs "$OUT"/main/* --heuristics greedy myopic --episodes 1024 --out "$OUT/results/main.csv"
python -m gtmarl.gate --res "$OUT/results" --proposed gippo
python -m gtmarl.gate --res "$OUT/results" --proposed gippo_msg | tail -8
