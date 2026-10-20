#!/usr/bin/env bash
# Phase 0 pilot (~2-2.5 h): 5 training runs in parallel (default setting of Sec. V-A),
# benchmarks, best-response gap and the go/no-go table.
# Usage: bash gtmarl/scripts/pilot.sh [OUT_DIR] [ITERS]
set -euo pipefail
OUT=${1:-/data/$USER/gtmarl_pilot}
ITERS=${2:-1500}
mkdir -p "$OUT/logs" "$OUT/results"
PY="nice -n 10 python -m gtmarl.train --iters $ITERS --train threads=1 device=cpu --out $OUT/main"
$PY --algo gippo    --seed 0 > "$OUT/logs/gippo_s0.log" 2>&1 &
$PY --algo gippo    --seed 1 > "$OUT/logs/gippo_s1.log" 2>&1 &
$PY --algo ippo_mlp --seed 0 > "$OUT/logs/ippo_mlp_s0.log" 2>&1 &
$PY --algo mappo    --seed 0 > "$OUT/logs/mappo_s0.log" 2>&1 &
$PY --algo gippo    --seed 0 --tag w0 --env w_fail_dist=0 > "$OUT/logs/gippo_w0.log" 2>&1 &   # v22 Eq. (21)
wait
python -m gtmarl.evaluate --runs "$OUT"/main/* --heuristics greedy myopic --episodes 1024 --out "$OUT/results/main.csv"
python -m gtmarl.br_gap --run "$OUT/main/gippo_N30_M4_s0" --iters 300 --out "$OUT/results/br_gippo_s0_last.json" &
python -m gtmarl.br_gap --run "$OUT/main/mappo_N30_M4_s0" --iters 300 --out "$OUT/results/br_mappo_s0_last.json" &
wait
python -m gtmarl.gate --res "$OUT/results"
