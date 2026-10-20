#!/usr/bin/env bash
# Sensitivity (optional, after the main sweep): penalty weight mu and failure grading w, gippo only.
# Usage: bash gtmarl/scripts/sens.sh [OUT_DIR] [MAX_PARALLEL]
set -euo pipefail
OUT=${1:-/data/$USER/gtmarl_runs}
P=${2:-16}
J="$OUT/sens_jobs.txt"; : > "$J"; mkdir -p "$OUT/logs"
PY="python -m gtmarl.train --iters ${ITERS:-1500} --train threads=1 device=cpu --out $OUT/sens"
for s in 0 1; do
  for mu in 0.5 1.0 1.5 4.0; do echo "$PY --algo gippo --seed $s --tag mu$mu --env mu=$mu" >> "$J"; done
  for w in 0 0.3; do echo "$PY --algo gippo --seed $s --tag w$w --env w_fail_dist=$w" >> "$J"; done
done
nl -nln "$J" | xargs -P "$P" -L 1 bash -c 'id=$0; shift; nice -n 10 "$@" > "'"$OUT"'/logs/sens_$id.log" 2>&1 || echo "FAILED $id"'
python -m gtmarl.evaluate --runs "$OUT"/sens/* "$OUT"/main/gippo_N20_M4_s0 "$OUT"/main/gippo_N20_M4_s1 \
       --episodes 1024 --out "$OUT/results/sens.csv"
