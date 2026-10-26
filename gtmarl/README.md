# GT-MARL for non-cooperative UAV swarm trajectory planning in LAWNs

Code for the TWC manuscript *Graph-Aided Game-Theoretic Multi-Agent Reinforcement
Learning for Non-Cooperative UAV Swarm Trajectory Planning in LAWNs* (**v22**).
Self-contained; it does not import `marltoolkit` (the ICCC code stays untouched).

| File | Content | Manuscript v22 |
|---|---|---|
| `config.py` | all parameters (`EnvConfig`, `TrainConfig`), SNR-level presets for L = 1..4 | Sec. II, V-A |
| `env.py` | vectorised Markov game G: association, average SNR, levels, bearers, ergodic Rician rate, demand, penalty, admissible set, cost, shaping, observations, metrics | Eqs. (1)-(29) |
| `models.py` | shared encoder f_ω (graph mean / graph with message MLP / k-NN MLP / none), actor, critic (MAPPO: V(s, o_i)), value normalisation | Eqs. (29)-(31) |
| `ppo.py` | graph-aided IPPO (γ = 1, GAE), IPPO-MLP, CTDE-MAPPO | Alg. 1, Eqs. (32)-(35) |
| `heuristics.py` | SP (shortest path), MBR (myopic best response) | V-A benchmarks |
| `br_gap.py` | normalised best-response gap ε̂ | Eq. (36) |
| `evaluate.py` | metrics, transfer across N, inference time, trajectories | V-C, V-D, V-G |
| `plots.py` | figures + LaTeX tables for V-B ... V-G | Sec. V |
| `gate.py` | go / no-go table after the pilot | |
| `tests/` | numerical checks of Prop. 1, Eq. (15), separation, shaping invariance | |

## Run (server)

```bash
cd ~/deep-marl-toolkit && git pull && git checkout gtmarl-twc
python -m pytest -q gtmarl/tests
bash gtmarl/scripts/pilot.sh gtmarl/data/gtmarl_pilot          # ~2-2.5 h, prints GO / NO-GO
bash gtmarl/scripts/pilot2.sh                                   # round 2: gippo_msg, ippo, fixed MAPPO, new BR gap
PROPOSED=gippo_msg bash gtmarl/scripts/sweep.sh ...             # choose the proposed variant after the pilot
bash gtmarl/scripts/sweep.sh gtmarl/data/gtmarl_runs 24         # 117 runs, ~10-12 h on 24 cores
bash gtmarl/scripts/post.sh  gtmarl/data/gtmarl_runs 16         # evaluation + BR gap, ~3 h
python -m gtmarl.plots --runs gtmarl/data/gtmarl_runs           # figs/*.pdf, *.tex
```
Single runs: `python -m gtmarl.train --algo gippo --seed 0 --env N=40 L=2 mu=3.0 --out runs/dev`.
CPU is the default and faster for these networks; one thread per process.

## Experiment → script → output

| Sec. | Experiment | Runs (sweep.sh) | Output (post.sh / plots.py) |
|---|---|---|---|
| V-B | return, ε̂, φ_x, entropy over training | `main/` | `fig_convergence.pdf` |
| V-C | SP, MBR, IPPO-MLP, CTDE-MAPPO, proposed | `main/` (5 seeds) | `table_main.tex` |
| V-D | N = 10..60, B ∈ {4, 9}; retrained and N = 30 policy without retraining; inference time | `scal/` | `fig_scalability.pdf`, `inference_time.csv` |
| V-E | L ∈ {1, 2, 3, 4}; μ ∈ {1, 2, 3, 6}·η | `quant/`, `mu/` | `fig_quant_mu.pdf` |
| V-F | w/o co-cell, w/o β_j, w/o q_i, w/o shaping (w_nav = 0, game unchanged), penalty β_i − κ_i, constant c_fail (w_fail = 0) | `abl/` | `table_ablation.tex` |
| V-G | SP vs proposed trajectories + cell utilisation | (main) | `fig_trajectories.pdf` |

## Learners (Oct 2026, obs_version 2)

All learners share the actor pi(a_i | o_i) and differ only in the neighbour encoder, the critic
input and the reward used by the critic/advantage (`ppo.ALGOS`):

| algo | neighbour encoder | critic | reward |
|---|---|---|---|
| `egat` (proposed) | 2-layer edge-featured graph attention over the local graph {i} U M_i (edges between neighbours too) | V(o_i) | own |
| `egat_aux` | + congestion head predicting D_b(t+1)/K_b (training only) | V(o_i) | own |
| `egat_cc` | as `egat` | agent-specific centralised V(x, o_i) | own (still non-cooperative) |
| `gippo` | mean of raw beacon features (v22 Eq. (29)) | V(o_i) | own |
| `gippo_msg` | message MLP, then mean (GraphSAGE-mean) | V(o_i) | own |
| `ippo_mlp` | k = 4 nearest beacons, sorted by distance | V(o_i) | own |
| `ippo` | none | V(o_i) | own |
| `mappo_egat` / `mappo` | egat / mean | centralised V(x, o_i) | team (social cost) |

Beacons (obs_version 2) carry position and velocity (Remote ID); the serving GBS of a neighbour
follows from its position. Defaults: N = 40, eta = 1.5, d_comm = 8 voxels (200 m).
Runs trained before this change load as obs_version 1. `bash gtmarl/scripts/night3.sh` runs the
encoder diagnostics, the MAPPO fix (critic warm-up) and zero-shot transfer.

## Default setting (Table I)

Values marked † are not fixed by v22 and were chosen here; please confirm.

| Parameter | Value |
|---|---|
| Area / corridor / grid | 1.5 × 1.5 km², [50, 150] m, 60 × 60 × 4 voxels (Δs = 25 m) |
| GBSs | B = 4 on a 2 × 2 grid, ISD 750 m, height 25 m†; B = 9 (ISD 500 m) in V-D |
| Carrier / RBs | 20 MHz, 30 kHz SCS → Δf = 360 kHz, K_b = 51 |
| Channel† | G_0 = −43.3 dB (3.5 GHz, 1 m), α = 2.3, Rician ϑ = 10 dB, per-RB power 6.5 dBm, σ² = −111.4 dBm/RB (NF 7 dB) → average SNR 12.0-37.4 dB |
| Levels / bearers† | L = 3: γ̄ = 25.6 / 17.1 / 10.0 dB → β = 2 / 3 / 5 RBs at R_min = 6 Mb/s (thresholds = inverse ergodic rate + 0.1 dB); presets for L = 1, 2, 4 in `config.py` |
| UAVs / origins | N = 30, three depots† at GBS 1-3 (radius 7 voxels, two lowest layers), destinations uniform, min. O-D distance 12 voxels† |
| Mobility† | v_max = 1 voxel/slot, d_safe = 1, d_comm = 4 voxels, η = 1.5 |
| Cost | μ = 4.5, c_fail = 100†, w_fail = 0.5† (graded failure cost, Eq. (21)); shaping w_nav = 0.5† (Eq. (25), with reset; Prop. 3) |
| PPO† | 32 episodes/iter, 1500 iters (1000 for V-D...V-F), lr 3e-4, clip 0.2, λ = 0.95, 5 epochs × 4 minibatches, entropy 0.01 → 0.001, 128-unit MLPs, d_e = 64, k = 4 for IPPO-MLP |

Episodes satisfy the route condition of Sec. II-A (initial and final configurations
overload no cell) whenever such a configuration is found; for large N at B = 4 the total
budget 4 × 51 RBs cannot host all bearers and the condition is relaxed. The share of
episodes meeting it is reported as `route_ok`.

## Metrics (`env.episode_metrics`, Sec. V-A)

`success` (MSR), `detour` (T_i/T_i^min of arrived UAVs), `p_out` (in-flight UAV-slots with
κ_i ρ(γ_i) < R_min), `jain_load` (Jain index of D_b/K_b), `social_cost` (Σ_i J_i), plus `J`,
`viol` (R_i < R_min, Eq. (9)), `potential` (φ_x per slot), `route_ok`, `infer_us`.

## Deviations from v22 to discuss

1. **Graded failure cost (on by default, `w_fail_dist` = 0.5).** With a constant c_fail and the
   deadline reset of the shaping, all failed trajectories have the same return, so a UAV that
   never arrives gets no learning signal (N = 1: MSR 0 with w_fail = 0, 0.95-0.98 with 0.5).
   Eq. (21) now reads c_fail + w_fail ||q_i - d_i||. `w_fail` is part of the game; `w_nav` is a
   learning hyperparameter, and Prop. 3 (checked in `tests/`) shows the shaped reward has the same
   Nash policies and an exact stage potential. The V-F row `nograde` sets w_fail = 0.
2. **Scale of c_fail.** One violating slot costs up to μβ = 4.5 × 5 ≈ 22, so with c_fail = 10 a UAV
   prefers to miss its deadline rather than cross a congested cell (an idle swarm scores a lower
   J than SP). c_fail = 100 restores the intended ordering.
3. **Benchmarks.** SP and MBR take a random admissible move after hovering two slots, otherwise
   they deadlock under the separation mask of Eq. (19).
