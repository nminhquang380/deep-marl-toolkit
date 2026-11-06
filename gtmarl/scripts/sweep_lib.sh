# Shared helpers of the sweep parts (sourced). Every part is resumable: re-running it skips finished
# training runs (--skip_done), existing BR-gap files and existing evaluation files; delete a file to redo it.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
IT=${IT:-1200}; TH_G=${TH_G:-2}; TX=${TX:-"n_envs=16 epochs=4"}
EV_EP=${EV_EP:-1024}
BR_ENVS=${BR_ENVS:-32}; BR_IT=${BR_IT:-100}; BR_VAL=${BR_VAL:-512}; BR_TEST=${BR_TEST:-2048}; BR_TH=${BR_TH:-4}
# GPU: DEV=cuda puts EGAT-based training, resumed runs, BR gaps and learned-policy evaluation on the GPUs
# listed in GPUS (round robin, one CUDA_VISIBLE_DEVICES per job); IPPO-MLP and heuristics stay on CPU.
DEV=${DEV:-cpu}; GPUS=(${GPUS:-1 2 3}); _GI=0
gpu() {  # sets GPU_ENV / GPU_ARG for the next job ("" on CPU)
  GPU_ENV=""; GPU_ARG=""; GPU_FLAG=""
  if [ "$DEV" = cuda ]; then
    GPU_ENV="CUDA_VISIBLE_DEVICES=${GPUS[$((_GI % ${#GPUS[@]}))]}"; GPU_ARG="device=cuda"; GPU_FLAG="--device cuda"
    _GI=$((_GI + 1))
  fi
}

setup() {  # OUT
  OUT=$1; L="$OUT/logs"; R="$OUT/results"; P="$R/parts"
  mkdir -p "$L" "$P"
  if pgrep -f "gtmarl.train .*--out $OUT/" > /dev/null; then
    echo "another run is already writing to $OUT; stop it first"; exit 1
  fi
  python -m pytest -q gtmarl/tests > "$L/tests.log" 2>&1 || { log "unit tests FAILED, see $L/tests.log"; exit 1; }
}
log() { echo "[$(date)] $*" | tee -a "$L/night.log"; }

# train SUBDIR ALGO SEED [TAG] [ENV overrides...]   (background job; EGAT-based algos get TH_G threads)
train() {
  local sub=$1 algo=$2 seed=$3 tag=${4:-}; shift 4 2>/dev/null || shift $#
  local th=1; GPU_ENV=""; GPU_ARG=""; case $algo in egat*|mappo_egat|gippo_cc) th=$TH_G; gpu;; esac
  local name="${sub}_${algo}_s${seed}${tag:+_$tag}"
  local envarg=""; [ $# -gt 0 ] && envarg="--env $*"
  env $GPU_ENV nice -n 10 python -m gtmarl.train --skip_done --train threads=$th $TX $GPU_ARG --iters $IT --algo $algo --seed $seed \
    ${tag:+--tag $tag} --out $OUT/$sub $envarg > $L/train_$name.log 2>&1 &
}
train_done() {
  wait
  log "training done"
  grep -l -E "Traceback|Error" $L/train_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"
  unfinished   # a killed process (OOM, admin) leaves no traceback
}
unfinished() {  # run directories without a final checkpoint; re-run the part to redo them
  for d in "$OUT"/*/*/; do
    [ -f "$d/config.json" ] && [ ! -f "$d/model_last.pt" ] && echo "UNFINISHED (killed?): $d" | tee -a "$L/night.log"
  done
  return 0
}

# evaluate NAME "RUN GLOBS" "HEURISTICS" "NS" [ENV overrides...]   (background; skipped if the csv exists)
evaluate() {
  local name=$1 runs=$2 heur=$3 ns=$4; shift 4
  local out=$P/$name.csv; [ -f "$out" ] && { log "skip evaluation $name (exists)"; return; }
  local args=""
  [ -n "$runs" ] && args="--runs $runs"
  [ -n "$heur" ] && args="$args --heuristics $heur"
  [ -n "$ns" ] && args="$args --N $ns"
  [ $# -gt 0 ] && args="$args --env $*"
  GPU_ENV=""; GPU_FLAG=""; [ -n "$runs" ] && gpu
  env $GPU_ENV nice -n 10 python -m gtmarl.evaluate $GPU_FLAG --episodes $EV_EP ${CKPT:+--ckpt $CKPT} $args --out $out \
    > $L/ev_$name.log 2>&1 &
}

# resume RUN_DIR ITERS   (continue a run, with its own configuration, to ITERS; entropy at its final value)
resume() {
  local run=$1 iters=$2
  [ -f "$run/model_last.pt" ] || { log "cannot resume $run (no checkpoint)"; return; }
  local th=1; GPU_ENV=""; GPU_ARG=""; case $(basename $run) in egat*|mappo_egat*|gippo_cc*) th=$TH_G; gpu;; esac
  env $GPU_ENV nice -n 10 python -m gtmarl.train --resume $run --skip_done --train threads=$th $TX $GPU_ARG ent_coef=0.001 \
    --iters $iters > $L/resume_$(basename $run).log 2>&1 &
}

# br RUN_DIR SEED NAME   (background; skipped if the json exists)
br() {
  local out=$R/br_$3.json
  [ -f "$out" ] && return
  [ -f "$1/model_last.pt" ] || { log "skip BR gap for $1 (no checkpoint)"; return; }
  gpu
  env $GPU_ENV nice -n 10 python -m gtmarl.br_gap $GPU_FLAG --threads $BR_TH --n_envs $BR_ENVS --val_episodes $BR_VAL \
    --eval_episodes $BR_TEST --iters $BR_IT --n_snap 2 --run "$1" --seed $2 --out $out > $L/br_$3.log 2>&1 &
}

# merge PREFIX -> results/PREFIX.csv from results/parts/PREFIX_*.csv
merge() {
  python - "$P" "$1" "$R/$1.csv" <<'PY'
import csv, glob, sys
rows = [r for f in sorted(glob.glob(f"{sys.argv[1]}/{sys.argv[2]}_*.csv")) for r in csv.DictReader(open(f))]
if rows:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
}

report() {  # SUBDIRS...
  { python -m gtmarl.summarize --res $R
    echo; echo "== learning curves"; python -m gtmarl.curves "$@" | sed "s|^|  |"; } > $OUT/summary.txt 2>&1
  cat $OUT/summary.txt
}
