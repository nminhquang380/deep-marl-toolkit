#!/usr/bin/env bash
# Part C extension (V-F): the highest-weight runs (marginal mu = 9, indicator mu = 22, shortfall mu = 200)
# were still improving at 1200 iterations (MSR 0.89 -> 0.97, 0.93 -> 0.96, 0.97 -> 0.94 over the last
# 200). Copies of all 11 part-C runs (pen1600/; the 1200-iteration runs and results/pen.csv are untouched)
# are continued to IT_LONG = 1600 iterations, the budget of the part-A main runs, so that the marginal
# mu = 4.5 reference of V-F is results/main.csv of part A (5 seeds, 1600 iterations).
# About 22 threads. Usage (inside tmux): bash gtmarl/scripts/sweepC_ext.sh    Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_LONG=${IT_LONG:-1600}
setup ${1:-$DATA/sweep/C}
log "part C extension: penalty runs to $IT_LONG iterations"
mkdir -p $OUT/pen1600
for src in $OUT/pen/*/; do
  dst=$OUT/pen1600/$(basename $src)
  [ -d "$dst" ] || cp -r "$src" "$dst"
  resume $dst $IT_LONG
done
wait
grep -l -E "Traceback|Error" $L/resume_*.log 2>/dev/null | sed 's/^/FAILED: /' | tee -a "$L/night.log"
log "part C extension: evaluation"
evaluate pen1600_all "$OUT/pen1600/*" "" ""
wait; merge pen1600
report $OUT/pen $OUT/pen1600
log "part C extension done"
