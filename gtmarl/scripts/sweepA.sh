#!/usr/bin/env bash
# Sweep part A (V-B, V-C): main comparison and the cooperative benchmark.
#   N = 50: egat_cc (proposed) and ippo_mlp, 5 seeds; transfer to N in {30, 40, 60}; SP, MBR
#   N = 30: egat_cc and mappo_egat (CTDE-MAPPO trains at this size), 2 seeds
#   BR gaps: egat_cc (N = 50, s0 s1), ippo_mlp (N = 50, s0), egat_cc and mappo_egat (N = 30, s0)
# Usage (inside tmux): bash gtmarl/scripts/sweepA.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/A}
log "part A: training (IT=$IT, TX=$TX)"
for s in 0 1 2 3 4; do train main egat_cc $s; train main ippo_mlp $s; done
for s in 0 1; do train n30 egat_cc $s "" N=30; train n30 mappo_egat $s "" N=30; done
train_done
log "part A: evaluation"
evaluate main_egat_cc "$OUT/main/egat_cc_N50_*" "" "30 40 50 60"
evaluate main_ippo_mlp "$OUT/main/ippo_mlp_N50_*" "" "30 40 50 60"
evaluate main_heur "" "greedy myopic" "30 40 50 60"
evaluate n30_all "$OUT/n30/*" "" "30"
wait; merge main; merge n30
report $OUT/main $OUT/n30
log "part A: BR gaps"
br $OUT/main/egat_cc_N50_M4_s0 0 egat_cc_s0
br $OUT/main/egat_cc_N50_M4_s1 1 egat_cc_s1
br $OUT/main/ippo_mlp_N50_M4_s0 0 ippo_mlp_s0
br $OUT/n30/egat_cc_N30_M4_s0 0 egat_cc_n30_s0
br $OUT/n30/mappo_egat_N30_M4_s0 0 mappo_egat_n30_s0
wait
report $OUT/main $OUT/n30
python -m gtmarl.gate --res $R --proposed egat_cc --N 50 | tee -a $OUT/summary.txt
log "part A done"
