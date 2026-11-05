#!/usr/bin/env bash
# GPU check before running the sweep with DEV=cuda (~15-20 min). Usage: bash gtmarl/scripts/gpu_check.sh [GPU]
# 1) can PyTorch use the GPU, 2) unit tests incl. the GPU equivalence of the torch environment,
# 3) 3 training iterations of egat_cc and ippo_mlp at N = 50 (n_envs 16, epochs 4) with
#    a) NumPy env + CPU networks, b) NumPy env + GPU networks, c) torch env + networks on the GPU,
# 4) a short BR gap and an evaluation with the torch env on the GPU.
set -uo pipefail
G=${1:-1}
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/gpucheck"; rm -rf $OUT; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== 1. CUDA"
CUDA_VISIBLE_DEVICES=$G python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" || exit 1
CUDA_VISIBLE_DEVICES=$G python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA not usable: stop here"; exit 1; }
echo "== 2. unit tests (torch env on CPU and GPU)"
CUDA_VISIBLE_DEVICES=$G python -m pytest -q gtmarl/tests 2>&1 | tail -1
echo "== 3. training speed (3 iterations, N = 50); runs one after the other"
TR="python -m gtmarl.train --iters 3 --train n_envs=16 epochs=4 eval_every=1000 save_every=3"
speed() {  # label seed algo env-and-device args
  local lab=$1 s=$2 algo=$3; shift 3
  CUDA_VISIBLE_DEVICES=$G $TR --algo $algo --seed $s --out $OUT "$@" > $OUT/$lab.log 2>&1 || { echo "  $lab FAILED"; tail -3 $OUT/$lab.log; return; }
  python - "$OUT/${algo}_N50_M4_s$s/progress.csv" "$lab" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); print(f"  {sys.argv[2]:32s} {float(r[-1]['time_min']) / len(r):.3f} min/iter")
PY
}
speed "egat_cc  numpy env + CPU"    11 egat_cc  --train threads=2
speed "egat_cc  numpy env + GPU"    12 egat_cc  --train threads=1 device=cuda
speed "egat_cc  torch env on GPU"   13 egat_cc  --train threads=1 device=cuda env_backend=torch
speed "ippo_mlp numpy env + CPU"    14 ippo_mlp --train threads=1
speed "ippo_mlp torch env on GPU"   15 ippo_mlp --train threads=1 device=cuda env_backend=torch
echo "== 4. BR gap and evaluation with the torch env on the GPU"
CUDA_VISIBLE_DEVICES=$G python -m gtmarl.br_gap --device cuda --backend torch --threads 1 \
  --run $OUT/egat_cc_N50_M4_s13 --iters 4 --n_snap 1 --n_envs 32 --val_episodes 64 --eval_episodes 64 \
  --out $OUT/br.json 2>&1 | grep -E "BR start|BR it|\"gap\"|Error" || exit 1
CUDA_VISIBLE_DEVICES=$G python -m gtmarl.evaluate --device cuda --backend torch --episodes 64 --n_envs 64 \
  --runs $OUT/egat_cc_N50_M4_s13 --out $OUT/ev.csv 2>&1 | grep -E "wrote|FAILED|Error"
echo "done: send the output. If c) is much faster than a), run the sweep with DEV=cuda GPUS=\"1 2 3\""
