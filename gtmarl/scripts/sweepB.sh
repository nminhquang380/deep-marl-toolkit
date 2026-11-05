#!/usr/bin/env bash
# Sweep part B (V-D, V-E deadline): retraining at other swarm sizes and deadline tolerances.
#   N in {40, 60}: egat_cc and ippo_mlp, 2 seeds, IT_SCAL = 1600 iterations as the part-A main comparison
#     (IPPO-MLP needs ~1200-1600 iterations to converge at N = 50); N = 50 comes from part A
#   eta in {1.2, 2.0, 3.0}: egat_cc, 2 seeds, IT_ETA = 1200 iterations; eta = 1.5 is the 1200-iteration
#     reference of part A (results/ref1200.csv); SP, MBR at every setting
# IT = 1200 (sweep_lib default). Usage (inside tmux): bash gtmarl/scripts/sweepB.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/B}
IT_SCAL=${IT_SCAL:-1600}; IT_ETA=${IT_ETA:-1200}
log "part B: training (IT_SCAL=$IT_SCAL, IT_ETA=$IT_ETA, TX=$TX, DEV=$DEV)"
for s in 0 1; do
  IT=$IT_SCAL
  for n in 40 60; do train scal egat_cc $s "" N=$n; train scal ippo_mlp $s "" N=$n; done
  IT=$IT_ETA
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
