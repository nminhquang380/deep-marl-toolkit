#!/usr/bin/env bash
# Sweep part D (V-F ablation, V-E rate requirement), relative to egat_cc of part A.
#   2 seeds: egat (local critic), gippo_cc (mean aggregation), egat_cc without neighbour-neighbour edges
#            (star), egat_cc without velocities in the beacons
#   seed 0:  egat_cc without shaping (w_nav = 0, game unchanged); egat_cc at R_min in {4, 8} Mb/s
# Usage (inside tmux): bash gtmarl/scripts/sweepD.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/D}
log "part D: training (IT=$IT, TX=$TX)"
for s in 0 1; do
  train abl egat $s; train abl gippo_cc $s
  train abl egat_cc $s star ablate=star; train abl egat_cc $s novel ablate=vel
done
train abl egat_cc 0 noshape w_nav=0
for r in 4e6 8e6; do train rmin egat_cc 0 R$r R_min=$r; done
train_done
log "part D: evaluation"
evaluate abl_all "$OUT/abl/*" "" ""
for r in 4e6 8e6; do evaluate rmin_$r "$OUT/rmin/*_R$r" "greedy myopic" "" R_min=$r; done
wait; merge abl; merge rmin
report $OUT/abl $OUT/rmin
log "part D done"
