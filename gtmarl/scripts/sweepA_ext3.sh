#!/usr/bin/env bash
# Part A, equal budget (Fig. 3, Table II): copies of the five EGAT-IPPO runs at N = 50 (main2400/; the
# 1600-iteration runs are untouched) are continued to IT_LONG = 2400 iterations, the budget of IPPO-MLP
# (sweepA_ext2.sh), and evaluated at N = 30-60 (results/egat2400.csv). The BR gap is then measured on the
# 2400-iteration checkpoints of both learners (egat_cc s0-s2, IPPO-MLP s0-s1).
# Training ~10 threads (~4 h), then 5 BR runs (~20 threads, ~2 h).
# Usage (inside tmux): bash gtmarl/scripts/sweepA_ext3.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_LONG=${IT_LONG:-2400}
setup ${1:-$DATA/sweep/A}
log "part A equal budget: EGAT-IPPO to $IT_LONG iterations"
mkdir -p $OUT/main2400
for s in 0 1 2 3 4; do
  src=$OUT/main/egat_cc_N50_M4_s$s; dst=$OUT/main2400/egat_cc_N50_M4_s$s
  [ -d "$dst" ] || cp -r "$src" "$dst"
  resume $dst $IT_LONG
done
wait
grep -l -E "Traceback|Error" $L/resume_egat_cc_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"
log "part A equal budget: evaluation"
evaluate egat2400_egat_cc "$OUT/main2400/egat_cc_N50_*" "" "30 40 50 60"
wait; merge egat2400
log "part A equal budget: BR gaps at $IT_LONG iterations"
for s in 0 1 2; do br $OUT/main2400/egat_cc_N50_M4_s$s $s egat_cc_${IT_LONG}_s$s; done
for s in 0 1; do br $OUT/main2400/ippo_mlp_N50_M4_s$s $s ippo_mlp_${IT_LONG}_s$s; done
wait
report $OUT/main $OUT/main2400 $OUT/n30 $OUT/mappo50
log "part A equal budget done"
