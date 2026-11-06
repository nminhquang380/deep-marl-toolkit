#!/usr/bin/env bash
# Part B extension (V-D): at N = 60 neither learner had converged at 1600 iterations (IPPO-MLP MSR
# 0.37 / 0.56 at 1200 -> 0.94 / 0.95 at 1600 with J not yet decreasing; egat_cc J still decreasing).
# Copies of the four N = 60 runs (scal2400/; the 1600-iteration runs and results/scal.csv are untouched)
# are continued to IT_LONG = 2400 iterations and evaluated at N = 60 (results/scal2400.csv).
# About 6 threads, so it can run alongside part C.
# Usage (inside tmux): bash gtmarl/scripts/sweepB_ext.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_LONG=${IT_LONG:-2400}
setup ${1:-$DATA/sweep/B}
log "part B extension: N = 60 runs to $IT_LONG iterations"
mkdir -p $OUT/scal2400
for s in 0 1; do
  for a in egat_cc ippo_mlp; do
    src=$OUT/scal/${a}_N60_M4_s$s; dst=$OUT/scal2400/${a}_N60_M4_s$s
    [ -d "$dst" ] || cp -r "$src" "$dst"
    resume $dst $IT_LONG
  done
done
wait
grep -l -E "Traceback|Error" $L/resume_*_N60_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"
log "part B extension: evaluation"
evaluate scal2400_N60 "$OUT/scal2400/*" "" "60"
wait; merge scal2400
report $OUT/scal $OUT/scal2400 $OUT/eta
log "part B extension done"
