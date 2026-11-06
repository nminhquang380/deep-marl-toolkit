#!/usr/bin/env bash
# Sweep part C (V-F penalty design): egat_cc with each penalty over a range of weights.
#   marginal (Eq. (14)) mu in {2.25, 9} (mu = 4.5: part A, results/ref1200.csv); lost RBs mu in {2.25, 4.5, 9};
#   indicator mu in {5.5, 11, 22}; normalised shortfall mu in {50, 100, 200}
#   Compare penalties through the P_out-detour trade-off (J is not comparable across penalties).
# Separation weight: w_sep = 30 must exceed the largest per-slot congestion penalty (mu * beta^(L) = 22.5 for
#   the marginal penalty at mu = 4.5). At the highest weight of each family this fails (marginal mu = 9:
#   up to 45 under SP), so w_sep is scaled with mu above the family's middle weight: w_sep = 60 at
#   mu = 9 / 9 / 22 / 200, i.e. w_sep / mu is kept as at the reference, and 30 otherwise.
# IT = 1200 (sweep_lib default). SEEDS (default "0"): e.g. SEEDS=1 adds a second seed in a later pass.
# Usage (inside tmux): bash gtmarl/scripts/sweepC.sh      Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
setup ${1:-$DATA/sweep/C}
SEEDS=${SEEDS:-0}
log "part C: training (IT=$IT, TX=$TX, seeds $SEEDS)"
for s in $SEEDS; do
  for m in 2.25 9; do w=30; [ $m = 9 ] && w=60; train pen egat_cc $s marginal_mu$m mu=$m w_sep=$w; done
  for m in 2.25 4.5 9; do w=30; [ $m = 9 ] && w=60; train pen egat_cc $s lost_mu$m penalty=lost mu=$m w_sep=$w; done
  for m in 5.5 11 22; do w=30; [ $m = 22 ] && w=60; train pen egat_cc $s indicator_mu$m penalty=indicator mu=$m w_sep=$w; done
  for m in 50 100 200; do w=30; [ $m = 200 ] && w=60; train pen egat_cc $s shortfall_mu$m penalty=shortfall mu=$m w_sep=$w; done
done
train_done
log "part C: evaluation"
evaluate pen_heur "" "greedy myopic" ""
for s in $SEEDS; do evaluate pen_s$s "$OUT/pen/*_s${s}_*" "" ""; done
wait; merge pen
report $OUT/pen
log "part C done"
