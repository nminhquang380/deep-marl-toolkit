#!/usr/bin/env bash
# Part A robustness checks for V-B / V-C (optional, run after or alongside part B):
#   1) IPPO-MLP convergence: its training cost was still decreasing at 1600 iterations, so copies of the
#      five part-A runs (main2400/, the 1600-iteration runs and their results are untouched) are continued
#      to IT_LONG = 2400 iterations and evaluated at N = 50 (results/ippo2400.csv).
#   2) More BR-gap seeds: egat_cc s2, IPPO-MLP s1, fine-tuned MAPPO s1 (1600-iteration checkpoints).
# Usage (inside tmux): bash gtmarl/scripts/sweepA_ext2.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_LONG=${IT_LONG:-2400}
setup ${1:-$DATA/sweep/A}
log "part A robustness: IPPO-MLP to $IT_LONG iterations; extra BR seeds"
mkdir -p $OUT/main2400
for s in 0 1 2 3 4; do
  src=$OUT/main/ippo_mlp_N50_M4_s$s; dst=$OUT/main2400/ippo_mlp_N50_M4_s$s
  [ -d "$dst" ] || cp -r "$src" "$dst"
  resume $dst $IT_LONG
done
br $OUT/main/egat_cc_N50_M4_s2 2 egat_cc_s2
br $OUT/main/ippo_mlp_N50_M4_s1 1 ippo_mlp_s1
br $OUT/mappo50/mappo_egat_N50_M4_s1_ft 1 mappo_egat_ft_s1
wait
grep -l -E "Traceback|Error" $L/resume_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"
log "part A robustness: evaluation"
evaluate ippo2400_ippo_mlp "$OUT/main2400/ippo_mlp_N50_*" "" "50"
wait; merge ippo2400
report $OUT/main $OUT/main2400 $OUT/n30 $OUT/mappo50
log "part A robustness done"
