#!/usr/bin/env bash
# Sweep part C (V-F penalty design): egat_cc with each penalty over a range of weights, seed 0.
#   marginal (Eq. (14)) mu in {2.25, 9} (mu = 4.5 from part A); lost RBs mu in {2.25, 4.5, 9};
#   indicator mu in {5.5, 11, 22}; normalised shortfall mu in {50, 100, 200}
#   Compare penalties through the P_out-detour trade-off (J is not comparable across penalties).
# IT = 1200 (sweep_lib default). Usage (inside tmux): bash gtmarl/scripts/sweepC.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/C}
log "part C: training (IT=$IT, TX=$TX)"
for m in 2.25 9; do train pen egat_cc 0 marginal_mu$m mu=$m; done
for m in 2.25 4.5 9; do train pen egat_cc 0 lost_mu$m penalty=lost mu=$m; done
for m in 5.5 11 22; do train pen egat_cc 0 indicator_mu$m penalty=indicator mu=$m; done
for m in 50 100 200; do train pen egat_cc 0 shortfall_mu$m penalty=shortfall mu=$m; done
train_done
log "part C: evaluation"
evaluate pen_all "$OUT/pen/*" "greedy myopic" ""
wait; merge pen
report $OUT/pen
log "part C done"
