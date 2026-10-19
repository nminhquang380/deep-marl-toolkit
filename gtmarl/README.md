# GT-MARL for non-cooperative UAV swarm trajectory planning in LAWNs

Code for the TWC manuscript *Graph-Aided Game-Theoretic Multi-Agent Reinforcement
Learning for Non-Cooperative UAV Swarm Trajectory Planning in LAWNs* (v18).
Self-contained; it does not import `marltoolkit` (the ICCC code stays untouched).

| File | Content | Manuscript |
|---|---|---|
| `config.py` | all parameters (`EnvConfig`, `TrainConfig`) | Sec. II, V |
| `env.py` | vectorised Markov game G: association, SNR levels, bearers, demand, penalty, admissible set, cost, shaping, metrics | Eqs. (1)-(10), (12), (15)-(19), (22), (24)-(28) |
| `models.py` | shared encoder f_ω, actor, critic, value normalisation | Eqs. (29)-(30) |
| `ppo.py` | graph-aided IPPO (γ = 1, GAE), IPPO w/o GNN, MAPPO (team reward, centralised critic) | Alg. 1, Eqs. (31)-(34) |
| `heuristics.py` | greedy shortest path, myopic best response | baselines |
| `br_gap.py` | empirical best-response (Nash) gap | Remark 1 |
| `evaluate.py` | metrics tables, scalability without retraining, trajectory dump | Sec. V |
| `plots.py` | IEEE-style figures + LaTeX table | Sec. V |
| `tests/` | numerical checks of Prop. 1, Eq. (13), separation, shaping invariance | |

## Setup (server)

```bash
conda create -n gtmarl python=3.11 -y && conda activate gtmarl
pip install torch numpy matplotlib pytest tensorboard    # CPU wheels are enough
cd deep-marl-toolkit && git checkout gtmarl-twc
python -m pytest -q gtmarl/tests                          # 6 tests, ~2 s
```

## Run

```bash
# single run (prints eval every 50 iterations; writes progress.csv, checkpoints)
python -m gtmarl.train --algo gippo --seed 0 --out runs/dev
python -m gtmarl.train --algo gippo --env N=30 K_b_scale_with_N=true --train lr=1e-4

# full sweep (48 runs) on 24 CPU cores, niced; then evaluation + Nash gap + figures
tmux new -s sweep
bash gtmarl/scripts/sweep.sh /data/$USER/gtmarl_runs 24
bash gtmarl/scripts/post.sh  /data/$USER/gtmarl_runs 16
python -m gtmarl.plots --runs /data/$USER/gtmarl_runs
```

CPU is the default and usually faster for these small networks. If a GPU is used,
pin one free card, e.g. `CUDA_VISIBLE_DEVICES=2 ... --device cuda`.

## Default setting (edit `config.py` or pass `--env key=value`)

| Parameter | Value |
|---|---|
| Airspace | 32 × 32 × 3 voxels, Δs = 40 m (1.28 km × 1.28 km), altitudes 60/100/140 m |
| GBSs | M = 4 on a 2 × 2 grid, height 25 m; second layout M = 9 (3 × 3), K_b = 10 |
| Radio | Δf = 180 kHz, effective per-RB power 0 dBm, β0 = −43.3 dB, α = 2.3, σ² = −112.4 dBm |
| SNR levels | thresholds 24/16/11/6 dB → bearers 1/2/3/4 RBs at R_min = 1.4 Mbps |
| RB budget | K_b = 22 (≈ 66 % mean load at N = 20); scaled with N for scalability |
| Mobility | v_max = 1, d_safe = 1, d_comm = 4, η = 1.5, min. O-D distance 8 voxels |
| Cost | μ = 2 (> η, Prop. 2), ρ_fail = 10, ρ_nav = 0.05 |
| PPO | 32 episodes/iter, lr 3e-4, clip 0.2, λ = 0.95, 5 epochs × 4 minibatches, entropy 0.01 → 0.001 |

Episodes are generated so that the initial and the final (all-arrived) configurations
overload no cell (operator-controlled deployment and the route condition of Sec. II-A),
and origins / destinations are pairwise at least d_safe + v_max apart.

## Metrics (`env.episode_metrics`)

`success` (arrival by H_i), `detour` (T_i / T_i^min of successful UAVs), `J` (mean
individual game cost, Eq. (20)), `welfare` (Σ_i J_i), `viol` (share of active slots with
R_i < R_min, Eq. (7)), `rate_ratio` (mean R_i / R_min), `jain_util` (Jain index of
T_i^min / T_i, 0 on failure), `jain_load` (Jain index of cell loads D_b / K_b),
`overload` (share of overloaded cell-slots).
