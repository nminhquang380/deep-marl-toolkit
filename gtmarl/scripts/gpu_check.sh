#!/usr/bin/env bash
# GPU check before running the sweep with DEV=cuda (~10-15 min). Usage: bash gtmarl/scripts/gpu_check.sh [GPU]
# 1) can PyTorch use the GPU, 2) 3 training iterations of egat_cc at N = 50 on GPU vs CPU,
# 3) a short BR gap and an evaluation on the GPU, 4) the unit tests (incl. the CUDA test).
set -uo pipefail
G=${1:-1}
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/gpucheck"; rm -rf $OUT; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== 1. CUDA"
CUDA_VISIBLE_DEVICES=$G python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" || exit 1
CUDA_VISIBLE_DEVICES=$G python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA not usable: stop here"; exit 1; }
echo "== 2. training speed (3 iterations, N = 50, n_envs 16, epochs 4)"
TR="python -m gtmarl.train --algo egat_cc --iters 3 --train n_envs=16 epochs=4 threads=2 eval_every=1000 save_every=3"
CUDA_VISIBLE_DEVICES=$G $TR device=cuda --seed 1 --out $OUT > $OUT/gpu.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $TR --seed 2 --out $OUT > $OUT/cpu.log 2>&1 &
wait
for s in 1 2; do
  python - "$OUT/egat_cc_N50_M4_s$s/progress.csv" "$([ $s = 1 ] && echo GPU || echo CPU)" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); print(f"  {sys.argv[2]}: {float(r[-1]['time_min']) / len(r):.2f} min/iter")
PY
done || { echo "training failed, see $OUT/gpu.log $OUT/cpu.log"; tail -5 $OUT/gpu.log; exit 1; }
echo "== 3. BR gap and evaluation on the GPU"
CUDA_VISIBLE_DEVICES=$G python -m gtmarl.br_gap --device cuda --threads 2 --run $OUT/egat_cc_N50_M4_s1 --iters 2 \
  --n_snap 1 --n_envs 8 --val_episodes 8 --eval_episodes 8 --out $OUT/br.json 2>&1 | grep -E "BR start|\"gap\"|Error" || exit 1
CUDA_VISIBLE_DEVICES=$G python -m gtmarl.evaluate --device cuda --episodes 16 --n_envs 16 --runs $OUT/egat_cc_N50_M4_s1 \
  --out $OUT/ev.csv 2>&1 | grep -E "wrote|FAILED|Error"
echo "== 4. unit tests"
CUDA_VISIBLE_DEVICES=$G python -m pytest -q gtmarl/tests 2>&1 | tail -1
echo "done: if 2-4 passed and the GPU is clearly faster, run the sweep parts with DEV=cuda GPUS=\"1 2 3\""
