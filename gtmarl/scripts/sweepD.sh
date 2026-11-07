#!/usr/bin/env bash
# Sweep part D (V-F ablation, V-E rate requirement), relative to egat_cc of part A.
#   2 seeds: egat (local critic), gippo_cc (mean aggregation), egat_cc without neighbour-neighbour edges
#            (star), egat_cc without velocities in the beacons
#   seed 0:  egat_cc without shaping (w_nav = 0, game unchanged); egat_cc at R_min in {4, 8} Mb/s
# IT = 1200 (sweep_lib default). Usage (inside tmux): bash gtmarl/scripts/sweepD.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/D}
log "part D: training (IT=$IT, TX=$TX)"
for s in 0 1; do
  train abl egat $s; train abl gippo_cc $s
  train abl egat_cc $s star ablate=star; train abl egat_cc $s novel ablate=vel
done
train abl egat_cc 0 noshape w_nav=0
# w_sep must exceed mu * beta^(L): beta^(L) = 4 / 5 / 7 RBs at R_min = 4 / 6 / 8 Mb/s, so mu * beta^(L) = 31.5 > 30
# at 8 Mb/s; w_sep = 42 keeps the reference margin (30 / 22.5 = 4/3)
train rmin egat_cc 0 R4e6 R_min=4e6
train rmin egat_cc 0 R8e6 R_min=8e6 w_sep=42
train_done
log "part D: evaluation"
evaluate abl_all "$OUT/abl/*" "" ""
evaluate rmin_4e6 "$OUT/rmin/*_R4e6" "greedy myopic" "" R_min=4e6
evaluate rmin_8e6 "$OUT/rmin/*_R8e6" "greedy myopic" "" R_min=8e6 w_sep=42
wait; merge abl; merge rmin
report $OUT/abl $OUT/rmin
log "part D done"
