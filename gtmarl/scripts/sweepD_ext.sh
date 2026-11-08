#!/usr/bin/env bash
# Part D extension (V-E rate requirement): trained from scratch at R_min = 8 Mb/s, egat_cc never learned to
# reach destinations (MSR 0.00 at every checkpoint; it hovers and fails at the deadline). The run is
# repeated warm-started from the 6 Mb/s policy (part A, egat_cc s0 / s1, 1600 iterations; encoder and actor,
# fresh critic with warm-up 200, lr 1e-4) for IT_FT = 800 iterations; same game (w_sep = 42).
# About 4 threads. Usage (inside tmux): bash gtmarl/scripts/sweepD_ext.sh    Re-run the same command to resume.
source "$(dirname "${BASH_SOURCE[0]}")/sweep_lib.sh"
IT_FT=${IT_FT:-800}
setup ${1:-$DATA/sweep/D}
A=$DATA/sweep/A/main
log "part D extension: R_min = 8 Mb/s warm-started from the 6 Mb/s policy ($IT_FT iterations)"
for s in 0 1; do
  WS=$A/egat_cc_N50_M4_s$s/model_1600.pt
  [ -f "$WS" ] || { log "missing $WS"; continue; }
  nice -n 10 python -m gtmarl.train --skip_done --train threads=$TH_G $TX init_from=$WS critic_warmup=200 \
    lr=1e-4 --iters $IT_FT --algo egat_cc --seed $s --tag R8e6_ft --out $OUT/rmin_ft \
    --env R_min=8e6 w_sep=42 > $L/train_rmin_ft_s$s.log 2>&1 &
done
train_done
log "part D extension: evaluation"
evaluate rminft_8e6 "$OUT/rmin_ft/*" "" "" R_min=8e6 w_sep=42
wait; merge rminft
report $OUT/abl $OUT/rmin $OUT/rmin_ft
log "part D extension done"
