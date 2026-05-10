#!/usr/bin/env bash
# Launches all 4 algorithms in a detached tmux session using 2 GPUs.
#   GPU 1: QMIX + IQL  (off-policy, small batches — low GPU memory pressure)
#   GPU 2: MAPPO + Random
#
# Re-attach later with:  tmux attach -t marl_exp
# Switch windows:        Ctrl-b  then  0 / 1 / 2 / 3

set -e

SESSION="marl_exp"
PYTHON="/home/ad/21503254/anaconda3/envs/MARL-LAE/bin/python"
TOTAL_STEPS=2000000
LOGGER="tensorboard"
ROOT="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT/logs"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session + 3 extra windows (4 total)
tmux new-session  -d -s "$SESSION" -n qmix  -x 220 -y 50
tmux new-window   -t "$SESSION"    -n iql
tmux new-window   -t "$SESSION"    -n mappo
tmux new-window   -t "$SESSION"    -n random

for win in qmix iql mappo random; do
    tmux send-keys -t "$SESSION:$win" "cd $ROOT" Enter
done

# ── GPU 1: QMIX ──────────────────────────────────────────────────────────────
tmux send-keys -t "$SESSION:qmix" \
  "CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/main_multi_uav_qmix.py \
    --logger $LOGGER --total_steps $TOTAL_STEPS \
    --train_log_interval 5 --test_log_interval 20 --num_test_episodes 32 \
    2>&1 | tee logs/qmix.log" Enter

# ── GPU 1: IQL ───────────────────────────────────────────────────────────────
tmux send-keys -t "$SESSION:iql" \
  "CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/main_multi_uav_iql.py \
    --logger $LOGGER --total_steps $TOTAL_STEPS \
    --train_log_interval 5 --test_log_interval 20 --num_test_episodes 32 \
    2>&1 | tee logs/iql.log" Enter

# ── GPU 2: MAPPO ─────────────────────────────────────────────────────────────
tmux send-keys -t "$SESSION:mappo" \
  "CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/main_multi_uav_mappo.py \
    --logger $LOGGER --total_steps $TOTAL_STEPS \
    --train_log_interval 5 --test_log_interval 20 --num_test_episodes 32 \
    2>&1 | tee logs/mappo.log" Enter

# ── GPU 2: Random (CPU baseline, CUDA_VISIBLE_DEVICES still set but unused) ──
tmux send-keys -t "$SESSION:random" \
  "CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/main_multi_uav_random.py \
    --logger $LOGGER --total_steps $TOTAL_STEPS \
    --train_log_interval 5 \
    2>&1 | tee logs/random.log" Enter

echo ""
echo "================================================"
echo " All 4 experiments launched → tmux '$SESSION'"
echo "================================================"
echo ""
echo "  tmux attach -t $SESSION            # re-attach"
echo "  Ctrl-b  0/1/2/3                    # switch windows"
echo "  tail -f logs/qmix.log              # follow a log"
echo "  tensorboard --logdir work_dirs/ --port 6006"
echo ""
