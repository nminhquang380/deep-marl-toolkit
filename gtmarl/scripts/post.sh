#!/usr/bin/env bash
# Evaluation, best-response gap and trajectories after sweep.sh (v22, Sec. V).
# Usage: bash gtmarl/scripts/post.sh [OUT_DIR (default gtmarl/data/...)] [MAX_PARALLEL]
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""   # CPU only: never touch the shared GPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"   # gtmarl/data
OUT=${1:-$DATA/gtmarl_runs}
P=${2:-16}
R=$OUT/results; mkdir -p "$R"
EV="nice -n 10 python -m gtmarl.evaluate --episodes 1024"
NS="10 20 30 40 50 60"
PA=${PROPOSED:-gippo}

# V-C main comparison (+ SP, MBR)
$EV --runs "$OUT"/main/* --heuristics greedy myopic --out "$R/main.csv"
# V-D scalability: transfer of the N = 30 policies, retrained policies, benchmarks
$EV --runs "$OUT"/main/${PA}_N30_M4_s* --N $NS --out "$R/scal_transfer_B4.csv"
$EV --runs "$OUT"/scal/${PA}_N30_M9_s* --N $NS --out "$R/scal_transfer_B9.csv"
$EV --runs "$OUT"/scal/* "$OUT"/main/${PA}_N30_M4_s* "$OUT"/main/mappo_N30_M4_s* --out "$R/scal_retrained.csv"
$EV --heuristics greedy myopic --N $NS --out "$R/scal_heur_B4.csv"
$EV --heuristics greedy myopic --N $NS --env M=9 --out "$R/scal_heur_B9.csv"
# V-E quantisation and penalty weight
$EV --runs "$OUT"/quant/* "$OUT"/main/${PA}_N30_M4_s* --out "$R/quant.csv"
for L in 1 2 3 4; do $EV --heuristics greedy myopic --env L=$L --out "$R/quant_heur_L$L.csv"; done
$EV --runs "$OUT"/mu/* "$OUT"/main/${PA}_N30_M4_s* --out "$R/mu.csv"
# V-F ablations
$EV --runs "$OUT"/abl/* "$OUT"/main/${PA}_N30_M4_s* --out "$R/abl.csv"
# V-G trajectories of SP and the proposed method on the same test episode
$EV --runs "$OUT"/main/${PA}_N30_M4_s0 --episodes 64 --traj "$R/traj_gippo.npz"
$EV --heuristics greedy --episodes 64 --traj "$R/traj_sp.npz"

# V-B / V-C best-response gap: checkpoints of the proposed method, final benchmarks
J=$R/br_jobs.txt; : > "$J"
for s in 0 1 2; do
  for c in 100 300 600 1000 1500; do
    echo "python -m gtmarl.br_gap --run $OUT/main/${PA}_N30_M4_s$s --ckpt model_$c.pt --out $R/br_${PA}_s${s}_$c.json" >> "$J"; done
  echo "python -m gtmarl.br_gap --run $OUT/main/mappo_N30_M4_s$s --ckpt model_1500.pt --out $R/br_mappo_s${s}_1500.json" >> "$J"
  echo "python -m gtmarl.br_gap --run $OUT/main/ippo_mlp_N30_M4_s$s --ckpt model_1500.pt --out $R/br_ippo_mlp_s${s}_1500.json" >> "$J"
done
xargs -P "$P" -L 1 nice -n 10 < "$J"
echo "results in $R"
