# Multi-UAV Trajectory Optimization — MARL Experiments

Cooperative multi-agent RL for joint trajectory planning and uplink rate
maximization, implemented on top of [deep-marl-toolkit](https://github.com/).

**System**: N=4 UAVs navigate a 10×10×3 voxel grid from random origins to
random destinations while maximizing uplink rate to M=4 ground base stations
(GBSs).  Agents share reward signal (cooperative).

---

## Installation

```bash
# 1. Create conda environment
conda create -n MARL-LAE python=3.11 -y
conda activate MARL-LAE

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install gymnasium tensorboard wandb numpy matplotlib tqdm

# 4. Install the toolkit (editable)
cd deep-marl-toolkit
pip install -e .
```

> **Note** – if `conda activate` is not available in your shell, use the
> direct Python path: `/home/ad/21503254/anaconda3/envs/MARL-LAE/bin/python`.
> All examples below assume the env is activated.

---

## Quick smoke-test

```bash
python -c "
from marltoolkit.envs.multi_uav import MultiUAVEnv
import numpy as np
env = MultiUAVEnv()
obs, state, _ = env.reset(seed=42)
print('obs:', obs.shape, '  state:', state.shape)
avail = env.get_available_actions()
actions = np.array([np.random.choice(np.where(a)[0]) for a in avail])
obs, state, r, term, trunc, info = env.step(actions)
print('reward:', r, '  arrived_rate:', env.arrived_rate)
"
```

---

## Running experiments

All scripts are in `scripts/` and run from the **repo root**.

### QMIX
```bash
python scripts/main_multi_uav_qmix.py --logger tensorboard
```

### IQL (independent Q-learning, no mixing network)
```bash
python scripts/main_multi_uav_iql.py --logger tensorboard
```

### MAPPO
```bash
python scripts/main_multi_uav_mappo.py --logger tensorboard
```

### Random baseline
```bash
python scripts/main_multi_uav_random.py --logger tensorboard
```

Logs and model checkpoints are saved under:
```
work_dirs/StarCraft2/multi_uav/<algo>/<timestamp>/
├── tensorboard_log_dir/   # TensorBoard events
├── models/
│   ├── ep_0000200/        # checkpoint at episode 200
│   │   ├── actor_model.th
│   │   └── optimizer.th
│   └── ep_0000400/
└── *.log                  # plaintext training log
```

---

## Key configuration options

### Common training flags (all scripts)
| Flag | Default | Description |
|------|---------|-------------|
| `--total_steps` | 10 000 000 | Total environment steps |
| `--logger` | `wandb` | `tensorboard` or `wandb` |
| `--batch_size` | 32 | Replay-buffer mini-batch size (QMIX/IQL) |
| `--memory_warmup_size` | 32 | Episodes before learning starts |
| `--train_log_interval` | 5 | Log every N episodes |
| `--test_log_interval` | 20 | Evaluate every N episodes |
| `--num_test_episodes` | 32 | Eval episodes per evaluation |
| `--log_dir` | `./work_dirs` | Root directory for all outputs |

### Environment reward weights (editable per-algo config or via config class)
These live in `configs/multi_uav_<algo>_config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `phi` | 0.5 | Navigation reward weight — higher → UAVs prioritise reaching destination |
| `rho_col` | 1.0 | Collision penalty magnitude |
| `r_fail` | 5.0 | Penalty for not arriving by T_max |
| `save_interval` | 200 | Save model checkpoint every N episodes |

---

## Experimenting with the reward trade-off (phi sweep)

`phi` controls the balance between **rate maximisation** and **navigation
speed**.  Edit the config class to run sweeps:

```python
# configs/multi_uav_qmix_config.py
phi: float = 0.0   # rate-only — UAVs hover near best GBS
phi: float = 0.25  # mild navigation pressure
phi: float = 0.5   # balanced (default)
phi: float = 1.0   # strong navigation pressure
```

Then run each setting separately and compare in TensorBoard:

```bash
# Rate-only
python -c "
import configs.multi_uav_qmix_config as cfg
cfg.MultiUAVQMixConfig.__init__.__defaults__ = (64,64,0.99,1.0,0.01,0.05,100,5e-4,1e-6,10.0,2,64,32,1,True,'qmix', 0.0, 1.0, 5.0, 200)
" && python scripts/main_multi_uav_qmix.py --logger tensorboard
```

Or more cleanly, duplicate the config file for each phi value:
```bash
cp configs/multi_uav_qmix_config.py configs/multi_uav_qmix_phi0_config.py
# Edit phi=0.0 in the copy, then import it in a duplicate script
```

A **recommended approach** for systematic sweeps: edit `phi` in the config,
rename `algo_name` so logs are separated (e.g., `algo_name='qmix_phi00'`), and
run each config.

---

## Viewing training curves

```bash
tensorboard --logdir work_dirs/ --port 6006
```

Open `http://localhost:6006`.  Key metrics logged per episode:

| Metric | Description |
|--------|-------------|
| `episode_reward` | Mean team reward |
| `win_rate` | Fraction of UAVs that arrived (0 = none, 1 = all) |
| `episode_step` | Steps taken (≤ T_max=100) |
| `loss` / `mean_td_error` | Q-learning loss (QMIX/IQL) |
| `policy_loss` / `value_loss` | PPO losses (MAPPO) |
| `exploration` | ε-greedy exploration rate (QMIX/IQL) |
| `learning_rate` | Current LR after decay |

---

## Visualizing trajectories

### Random policy (no checkpoint needed)
```bash
python scripts/visualize_trajectories.py --algo random --num_episodes 3 \
    --out figures/random_traj.pdf
```

### Trained QMIX / IQL
```bash
python scripts/visualize_trajectories.py \
    --algo qmix \
    --model_path work_dirs/StarCraft2/multi_uav/qmix/<timestamp>/models/ep_0005000 \
    --num_episodes 3 \
    --out figures/qmix_traj.pdf
```

### Compare policy evolution across training stages
```bash
python scripts/visualize_trajectories.py \
    --algo qmix \
    --checkpoints \
        work_dirs/.../models/ep_0000200 \
        work_dirs/.../models/ep_0002000 \
        work_dirs/.../models/ep_0010000 \
    --seed 7 \
    --out figures/qmix_evolution.pdf
```

Each subplot shows **○ = start**, **★ = end position**, **▲ = GBS**.
Arrival rate and episode reward are shown in the subplot title.

### Override environment settings at viz time
```bash
python scripts/visualize_trajectories.py --algo qmix \
    --model_path ... --phi 0.0 --num_episodes 4
```

---

## Config reference

### `configs/multi_uav_qmix_config.py`
```
fc_hidden_dim, rnn_hidden_dim   Actor network widths
mixing_embed_dim                QMIX mixer embed dimension
hypernet_layers, hypernet_embed Hypernetwork architecture
gamma                           Discount factor
learning_rate / min_learning_rate
egreedy_exploration             Initial epsilon
min_exploration                 Final epsilon
target_update_tau               Soft-update coefficient
target_update_interval          Steps between target updates
learner_update_freq             Gradient steps per episode
clip_grad_norm                  Gradient clipping
double_q                        Double-DQN
phi / rho_col / r_fail          Reward shaping (see above)
save_interval                   Checkpoint frequency (episodes)
```

### `configs/multi_uav_mappo_config.py`
```
hidden_size         MLP hidden units for actor and critic
ppo_epoch           PPO update epochs per episode
num_mini_batch      Minibatches per PPO epoch
clip_param          PPO clip ε
entropy_coef        Entropy bonus coefficient
value_loss_coef     Critic loss weight
max_grad_norm       Gradient clip
actor_lr / critic_lr
gamma / gae_lambda
use_valuenorm       Running-mean value normalisation (recommended)
use_huber_loss      Huber critic loss
phi / rho_col / r_fail
save_interval
```

---

## Environment details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid | 10×10×3 | X × Y × Z voxels |
| N agents | 4 | |
| M GBSs | 4 | Fixed at grid corners, z=0 |
| Action space | 25 discrete | L1 displacements ≤ v_max=2, index 0 = stay |
| Observation dim | 24 | Local obs per UAV (position, destination, T_min, R_bar, GBS load, GBS positions) |
| Actor input dim | 53 | Augmented: obs + last_action one-hot (25) + agent_id one-hot (4) |
| State dim | 48 | Global state for centralized critic |
| T_max | 100 | Episode length |
| Bandwidth per GBS | K_b=10 RBs × Δf=180 kHz | RBs split equally among co-served UAVs |
| Path loss | α=2.0, β₀=1e-3 | Free-space with floor |
| Tx power | 23 dBm ≈ 0.2 W | |

### Reward formula (per agent, averaged to scalar team reward)

```
r_rate = (R̄_i(t) - R̄_i(t-1)) / R_norm          # EMA rate increment
r_nav  = phi × (d(q_old, dest) - d(q_new, dest)) # navigation potential
r_col  = -rho_col  if collision else 0
r_term = R̄_i * T_min / T_i   if just arrived     # efficiency bonus
       = -r_fail               if T≥T_max & not arrived
```

`R_norm` = maximum single-UAV rate at 1-voxel distance (≈ normalisation
constant so rate rewards are O(1)).

---

## Recommended experiment plan

| Phase | Steps | Time est. | Purpose |
|-------|-------|-----------|---------|
| Sanity check | 50 K | ~3 min | Verify env + logging work |
| Short run | 500 K | ~30 min | Check learning is happening |
| Paper run | 2–5 M | 2–8 hr | Publication-quality curves |
| Phi sweep (×5) | 1 M each | ~3 hr | Trade-off analysis |

Run all four algorithms in **parallel** on separate GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/main_multi_uav_qmix.py  --logger tensorboard &
CUDA_VISIBLE_DEVICES=1 python scripts/main_multi_uav_iql.py   --logger tensorboard &
CUDA_VISIBLE_DEVICES=2 python scripts/main_multi_uav_mappo.py --logger tensorboard &
CUDA_VISIBLE_DEVICES=3 python scripts/main_multi_uav_random.py --logger tensorboard &
wait
```
