#!/usr/bin/env bash
# Sweep part B (V-D, V-E deadline): retraining at other swarm sizes and deadline tolerances.
#   N in {40, 60}: egat_cc and ippo_mlp, 2 seeds (N = 30, 50 come from part A)
#   eta in {1.2, 2.0, 3.0}: egat_cc, 2 seeds (eta = 1.5 comes from part A); SP, MBR at every setting
# Usage (inside tmux): bash gtmarl/scripts/sweepB.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/B}
log "part B: training (IT=$IT, TX=$TX)"
for s in 0 1; do
  for n in 40 60; do train scal egat_cc $s "" N=$n; train scal ippo_mlp $s "" N=$n; done
  for e in 1.2 2.0 3.0; do train eta egat_cc $s eta$e eta=$e; done
done
train_done
log "part B: evaluation"
for n in 40 60; do
  evaluate scal_N$n "$OUT/scal/*_N${n}_*" "greedy myopic" "$n"
done
for e in 1.2 2.0 3.0; do
  evaluate eta_$e "$OUT/eta/*_eta$e" "greedy myopic" "" eta=$e
done
wait; merge scal; merge eta
report $OUT/scal $OUT/eta
log "part B done"
