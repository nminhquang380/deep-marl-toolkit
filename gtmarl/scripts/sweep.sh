#!/usr/bin/env bash
# All training runs of Sec. V (v22). CPU only, one thread per process, niced.
# Usage: bash gtmarl/scripts/sweep.sh [OUT_DIR (default gtmarl/data/...)] [MAX_PARALLEL]
#   ITERS (main, default 1500), ITERS_AUX (other experiments, default 1000)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""   # CPU only: never touch the shared GPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"   # gtmarl/data
OUT=${1:-$DATA/gtmarl_runs}
P=${2:-24}
ITERS=${ITERS:-1500}
IA=${ITERS_AUX:-1000}
S5=${SEEDS_MAIN:-"0 1 2 3 4"}
S3=${SEEDS_AUX:-"0 1 2"}
PA=${PROPOSED:-gippo}            # proposed learner: gippo (v22 Eq. (29)) or gippo_msg
mkdir -p "$OUT/logs"
J="$OUT/jobs.txt"; : > "$J"
PY="python -m gtmarl.train --train threads=1 device=cpu"

# V-D scalability (longest first): retrained per N; N = 30, B = 4 comes from V-C
for N in 60 50 40 20 10; do for s in $S3; do for a in $PA mappo; do
  echo "$PY --iters $IA --algo $a --seed $s --out $OUT/scal --env N=$N M=4" >> "$J"; done; done; done
for N in 60 50 40 30 20 10; do for s in $S3; do for a in $PA mappo; do
  echo "$PY --iters $IA --algo $a --seed $s --out $OUT/scal --env N=$N M=9" >> "$J"; done; done; done
# V-B / V-C main comparison (N = 30, B = 4)
for s in $S5; do for a in $PA ippo_mlp mappo; do
  echo "$PY --iters $ITERS --algo $a --seed $s --out $OUT/main" >> "$J"; done; done
# V-E SNR quantisation L and penalty weight mu (L = 3, mu = 3 eta = 4.5 come from V-C)
for s in $S3; do
  for L in 1 2 4;      do echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/quant --tag L$L --env L=$L" >> "$J"; done
  for mu in 1.5 3.0 9.0; do echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/mu --tag mu$mu --env mu=$mu" >> "$J"; done
done
# V-F ablations
for s in $S3; do
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag nococell --env ablate=cocell"   >> "$J"
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag nobeta   --env ablate=beta_nbr" >> "$J"
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag nopos    --env ablate=own_pos"  >> "$J"
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag noshape  --env w_nav=0"         >> "$J"
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag lostpen  --env penalty=lost"    >> "$J"
  echo "$PY --iters $IA --algo $PA --seed $s --out $OUT/abl --tag graded   --env w_fail_dist=0.5" >> "$J"
  if [ "$PA" = gippo_msg ]; then echo "$PY --iters $IA --algo gippo --seed $s --out $OUT/abl --tag rawmean" >> "$J"; fi
done

echo "$(wc -l < "$J") jobs -> $OUT  (parallel $P)"
nl -nln "$J" | xargs -P "$P" -L 1 bash -c 'id=$0; shift; nice -n 10 "$@" > "'"$OUT"'/logs/job_$id.log" 2>&1 || echo "FAILED job $id: $*"'
echo "all done"
