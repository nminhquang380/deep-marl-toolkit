#!/usr/bin/env bash
# Post-processing after sweep.sh: evaluation tables, scalability, Nash gap, trajectories.
set -euo pipefail
OUT=${1:-/data/$USER/gtmarl_runs}
P=${2:-16}
R=$OUT/results; mkdir -p "$R"
EV="nice -n 10 python -m gtmarl.evaluate --episodes 1024"

# main table (N = 20, M = 4) incl. heuristics
$EV --runs "$OUT"/main/*_s* --heuristics greedy myopic --out "$R/main.csv"
# scalability: N=20 models evaluated at all N without retraining + per-N retrained models
$EV --runs "$OUT"/main/gippo_N20_M4_s* "$OUT"/main/ippo_N20_M4_s* --N 10 20 30 40 50 \
    --env K_b_scale_with_N=true --out "$R/scal_transfer.csv"
$EV --runs "$OUT"/scal/* "$OUT"/main/mappo_N20_M4_s* --env K_b_scale_with_N=true --out "$R/scal_retrained.csv"
$EV --heuristics greedy myopic --N 10 20 30 40 50 --env K_b_scale_with_N=true --out "$R/scal_heur.csv"
# second GBS configuration
$EV --runs "$OUT"/m9/* --heuristics greedy myopic --env M=9 K_b=10 --out "$R/m9.csv"
# trajectories for the visualisation figure
$EV --runs "$OUT"/main/gippo_N20_M4_s0 --episodes 64 --traj "$R/traj_gippo.npz"
$EV --runs "$OUT"/main/mappo_N20_M4_s0 --episodes 64 --traj "$R/traj_mappo.npz"

# Nash gap vs training iteration (seed 0..2) and for the final MAPPO policy
J=$R/br_jobs.txt; : > "$J"
for s in 0 1 2; do for c in 100 300 600 1000 1500; do
  echo "python -m gtmarl.br_gap --run $OUT/main/gippo_N20_M4_s$s --ckpt model_$c.pt --out $R/br_gippo_s${s}_$c.json" >> "$J"; done
  echo "python -m gtmarl.br_gap --run $OUT/main/mappo_N20_M4_s$s --ckpt model_1500.pt --out $R/br_mappo_s${s}_1500.json" >> "$J"
  echo "python -m gtmarl.br_gap --run $OUT/main/ippo_N20_M4_s$s --ckpt model_1500.pt --out $R/br_ippo_s${s}_1500.json" >> "$J"
done
xargs -P "$P" -L 1 nice -n 10 < "$J"
echo "results in $R"
