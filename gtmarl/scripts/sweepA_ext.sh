#!/usr/bin/env bash
# Part A extension: IPPO-MLP had not converged at 800 iterations, so the main runs of part A are
# continued to IT_EXT = 1600 (egat_cc too, for an equal budget; entropy kept at its final value), and
# CTDE-MAPPO, which does not learn from scratch within the budget, is fine-tuned on the social cost
# from the 800-iteration egat_cc policies (critic warm-up 200, lr 5e-5) at N = 50.
# The 800-iteration evaluation and BR gaps are kept in results/it800. The 1200-iteration egat_cc
# checkpoints are also evaluated (results/ref1200.csv): parts B, C, D train for 1200 iterations.
# Usage (inside tmux): bash gtmarl/scripts/sweepA_ext.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_EXT=${IT_EXT:-1600}
setup ${1:-$DATA/sweep/A}
log "part A extension: continue to $IT_EXT iterations; MAPPO fine-tuning"
for s in 0 1 2 3 4; do
  resume $OUT/main/egat_cc_N50_M4_s$s $IT_EXT; resume $OUT/main/ippo_mlp_N50_M4_s$s $IT_EXT
done
for s in 0 1; do
  WS=$OUT/main/egat_cc_N50_M4_s$s/model_800.pt
  nice -n 10 python -m gtmarl.train --skip_done --train threads=$TH_G $TX init_from=$WS critic_warmup=200 \
    lr=5e-5 --iters 800 --algo mappo_egat --seed $s --tag ft --out $OUT/mappo50 \
    > $L/train_mappo50_s$s.log 2>&1 &
done
wait
log "part A extension: training done"
grep -l -E "Traceback|Error" $L/resume_*.log $L/train_mappo50_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"

mkdir -p $R/it800
for f in $P/main_egat_cc.csv $P/main_ippo_mlp.csv $R/br_egat_cc_s0.json $R/br_egat_cc_s1.json $R/br_ippo_mlp_s0.json; do
  [ -f "$f" ] && [ ! -f "$R/it800/$(basename $f)" ] && mv "$f" $R/it800/
done
log "part A extension: evaluation"
evaluate main_egat_cc "$OUT/main/egat_cc_N50_*" "" "30 40 50 60"
evaluate main_ippo_mlp "$OUT/main/ippo_mlp_N50_*" "" "30 40 50 60"
evaluate mappo50_all "$OUT/mappo50/*" "" "50"
CKPT=model_1200.pt evaluate ref1200_egat_cc "$OUT/main/egat_cc_N50_*" "" "50"
wait; merge main; merge mappo50; merge ref1200
report $OUT/main $OUT/n30 $OUT/mappo50
log "part A extension: BR gaps"
br $OUT/main/egat_cc_N50_M4_s0 0 egat_cc_s0
br $OUT/main/egat_cc_N50_M4_s1 1 egat_cc_s1
br $OUT/main/ippo_mlp_N50_M4_s0 0 ippo_mlp_s0
br $OUT/mappo50/mappo_egat_N50_M4_s0_ft 0 mappo_egat_ft_s0
wait
report $OUT/main $OUT/n30 $OUT/mappo50
python -m gtmarl.gate --res $R --proposed egat_cc --N 50 | tee -a $OUT/summary.txt
log "part A extension done"
