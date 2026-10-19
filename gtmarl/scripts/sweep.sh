#!/usr/bin/env bash
# Full experiment sweep for the TWC paper (Sec. V).
# Usage:  bash gtmarl/scripts/sweep.sh [OUT_DIR] [MAX_PARALLEL]
# Runs on CPU only, one thread per process, niced, so the shared server stays usable.
set -euo pipefail
OUT=${1:-/data/$USER/gtmarl_runs}
P=${2:-24}
ITERS=${ITERS:-1500}
SEEDS_MAIN=${SEEDS_MAIN:-"0 1 2 3 4"}
SEEDS_AUX=${SEEDS_AUX:-"0 1 2"}
mkdir -p "$OUT/logs"
JOBS="$OUT/jobs.txt"; : > "$JOBS"
PY="python -m gtmarl.train --iters $ITERS --train threads=1 device=cpu"

# (2) [longest jobs first] scalability: retrain per N (load factor kept by scaling K_b); gippo is also
#     evaluated without retraining from the N = 20 model
for N in 50 40 30 10; do for s in $SEEDS_AUX; do for a in gippo mappo; do
  echo "$PY --algo $a --seed $s --out $OUT/scal --env N=$N K_b_scale_with_N=true" >> "$JOBS"; done; done; done
# (1) main comparison, N = 20, M = 4
for s in $SEEDS_MAIN; do for a in gippo ippo mappo; do
  echo "$PY --algo $a --seed $s --out $OUT/main" >> "$JOBS"; done; done
# (3) second GBS configuration: M = 9 on a 3x3 grid, K_b = 12 matches the congestion level of M = 4 (greedy violation ~11%)
for s in $SEEDS_AUX; do for a in gippo ippo mappo; do
  echo "$PY --algo $a --seed $s --out $OUT/m9 --env M=9 K_b=12" >> "$JOBS"; done; done

echo "$(wc -l < "$JOBS") jobs -> $OUT  (parallel $P)"
nl -nln "$JOBS" | \
  xargs -P "$P" -L 1 bash -c 'id=$0; shift; nice -n 10 "$@" > "'"$OUT"'/logs/job_$id.log" 2>&1 || echo "FAILED job $id: $*"'
echo "all done"
