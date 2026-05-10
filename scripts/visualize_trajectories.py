"""Visualize UAV trajectories from a saved model checkpoint.

Usage examples
--------------
# Random policy (no model needed):
python scripts/visualize_trajectories.py --algo random --num_episodes 3

# Trained QMIX / IQL checkpoint:
python scripts/visualize_trajectories.py --algo qmix \
    --model_path work_dirs/StarCraft2/multi_uav/qmix/<run>/models/ep_0001000 \
    --num_episodes 3

# Trained MAPPO checkpoint:
python scripts/visualize_trajectories.py --algo mappo \
    --model_path work_dirs/StarCraft2/multi_uav/mappo/<run>/models/ep_0000050 \
    --num_episodes 3

# Compare multiple checkpoints side-by-side (QMIX):
python scripts/visualize_trajectories.py --algo qmix \
    --checkpoints \
        work_dirs/.../models/ep_0000200 \
        work_dirs/.../models/ep_0001000 \
        work_dirs/.../models/ep_0005000 \
    --num_episodes 1

Optional flags
--------------
--phi 0.3           override environment phi
--out figures/traj.pdf   save figure (default: show interactively)
--seed 42
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # use non-interactive backend; overridden if --out not set
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marltoolkit.envs.multi_uav.multi_uav_env import MultiUAVEnv


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']


def _build_env(cli_args: argparse.Namespace, augment_obs: bool = False) -> MultiUAVEnv:
    """Build env; augment_obs=True adds last-action + agent-id to observations."""
    env_kwargs = {}
    for key in ('phi', 'rho_col', 'r_fail', 'v_max', 'T_max'):
        if getattr(cli_args, key, None) is not None:
            env_kwargs[key] = getattr(cli_args, key)
    env = MultiUAVEnv(**env_kwargs)
    if augment_obs:
        env.args = argparse.Namespace(use_last_actions=True,
                                      use_agents_id_onehot=True,
                                      use_global_state=False)
    return env


def _run_episode_random(env: MultiUAVEnv, seed: int = None):
    obs, state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        avail = env.get_available_actions()
        actions = np.array([
            np.random.choice(np.where(avail[i])[0])
            for i in range(env.num_agents)
        ])
        _, _, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        total_reward += reward
    return env.get_trajectory(), env.arrived_rate, total_reward


def _run_episode_qmix_iql(env: MultiUAVEnv, agent, seed: int = None):
    import torch
    agent.init_hidden_states(batch_size=1)
    obs, state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        avail = env.get_available_actions()
        actions = agent.predict(obs=obs, available_actions=avail)
        obs, state, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        total_reward += reward
    return env.get_trajectory(), env.arrived_rate, total_reward


def _run_episode_mappo(env: MultiUAVEnv, agent, device, seed: int = None):
    import torch
    agent.prep_rollout()
    obs, state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    actor_rnn = None
    while not done:
        avail = env.get_available_actions()
        obs_t = torch.FloatTensor(obs).to(device)
        state_t = torch.FloatTensor(state).unsqueeze(0).expand(
            env.num_agents, -1).contiguous().to(device)
        masks_t = torch.ones(env.num_agents, 1).to(device)
        avail_t = torch.FloatTensor(avail).to(device)
        with torch.no_grad():
            actions, _, actor_rnn = agent.actor_model.forward(
                obs_t, masks_t, avail_t, actor_rnn, deterministic=True)
        actions_np = actions.cpu().numpy().flatten()
        obs, state, reward, terminated, truncated, _ = env.step(actions_np)
        done = terminated or truncated
        total_reward += reward
    return env.get_trajectory(), env.arrived_rate, total_reward


def _load_qmix_agent(env: MultiUAVEnv, model_path: str, algo: str,
                     args_extra: argparse.Namespace):
    import torch
    import argparse as ap
    from marltoolkit.modules.actors import RNNActorModel
    from marltoolkit.agents.qmix_agent import QMixAgent

    # Build minimal args — must match training config augmentation flags
    # env.args already has use_last_actions/use_agents_id_onehot set by _build_env
    a = ap.Namespace()
    a.use_last_actions = True
    a.use_agents_id_onehot = True
    a.use_global_state = False
    a.fc_hidden_dim = 64
    a.rnn_hidden_dim = 64
    a.n_actions = env.n_actions
    a.obs_shape = env.get_actor_input_shape()

    actor = RNNActorModel(a)
    device = torch.device('cpu')
    agent = QMixAgent(
        actor_model=actor, mixer_model=None,
        num_agents=env.num_agents, double_q=True, total_steps=1,
        gamma=0.99, learning_rate=5e-4, min_learning_rate=1e-6,
        egreedy_exploration=0.0, min_exploration=0.0,
        target_update_tau=0.05, target_update_interval=100,
        learner_update_freq=1, clip_grad_norm=10.0, device=device,
    )
    actor_path = os.path.join(model_path, 'actor_model.th')
    agent.actor_model.load_state_dict(
        torch.load(actor_path, map_location=device))
    agent.actor_model.eval()
    return agent, device


def _load_mappo_agent(env: MultiUAVEnv, model_path: str,
                      args_extra: argparse.Namespace):
    import torch
    import gymnasium as gym
    import argparse as ap
    from marltoolkit.agents.mappo_agent import MAPPOAgent

    a = ap.Namespace(
        device=torch.device('cpu'),
        actor_input_dim=env.get_actor_input_shape()[0],
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        action_space=gym.spaces.Discrete(env.n_actions),
        hidden_size=128, rnn_layers=1,
        use_recurrent_policy=False, use_naive_recurrent_policy=False,
        use_orthogonal=True, use_feature_normalization=True,
        activation='relu', gain=0.01, data_chunk_length=10,
        ppo_epoch=10, num_mini_batch=4, clip_param=0.2,
        entropy_coef=0.01, value_loss_coef=1.0, max_grad_norm=10.0,
        use_max_grad_norm=True, use_clipped_value_loss=True,
        use_huber_loss=True, huber_delta=10.0,
        use_value_active_masks=True, use_policy_active_masks=True,
        use_popart=False, use_valuenorm=True,
        actor_lr=5e-4, critic_lr=5e-4, opti_eps=1e-5, weight_decay=0.0,
        algorithm_name='mappo',
    )
    agent = MAPPOAgent(a)
    actor_path = os.path.join(model_path, 'actor_model.pt')
    agent.actor_model.load_state_dict(
        torch.load(actor_path, map_location=torch.device('cpu')))
    agent.actor_model.eval()
    return agent, torch.device('cpu')


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectories(trajectories_list, env: MultiUAVEnv,
                      subplot_titles=None, fig_title='UAV Trajectories',
                      out_path: str = None):
    """
    Parameters
    ----------
    trajectories_list : list of (trajectory, arrived_rate, reward)
        Each item corresponds to one subplot column.
        trajectory : np.ndarray (T+1, N, 3)
    env : MultiUAVEnv  (for destinations and GBS positions)
    subplot_titles : list of str, one per subplot
    """
    n_plots = len(trajectories_list)
    fig = plt.figure(figsize=(6 * n_plots, 6))
    fig.suptitle(fig_title, fontsize=14, y=1.01)

    N = env.num_agents
    # Use the LAST episode's destinations/origins for annotation
    # (they differ per episode; we annotate the first/only one)
    traj_sample, arrived_rate, reward = trajectories_list[0]
    origins = traj_sample[0]           # (N, 3)
    # destinations not directly accessible after reset; annotate from traj end
    # (UAVs that arrived are at destination in the last step)

    for idx, (traj, arr_rate, ep_reward) in enumerate(trajectories_list):
        ax = fig.add_subplot(1, n_plots, idx + 1, projection='3d')

        T = traj.shape[0] - 1   # number of steps

        for i in range(N):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            xs = traj[:, i, 0].astype(float)
            ys = traj[:, i, 1].astype(float)
            zs = traj[:, i, 2].astype(float)

            # trajectory line
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.8,
                    label=f'UAV {i}')
            # origin marker
            ax.scatter(xs[0], ys[0], zs[0], color=color, marker='o',
                       s=80, zorder=5, edgecolors='k', linewidths=0.5)
            # final position marker (destination if arrived)
            ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='*',
                       s=120, zorder=5, edgecolors='k', linewidths=0.5)

        # GBS positions
        for b, gbs in enumerate(env.gbs_positions):
            ax.scatter(gbs[0], gbs[1], gbs[2], color='black', marker='^',
                       s=100, zorder=6,
                       label='GBS' if b == 0 else None)
            ax.text(gbs[0], gbs[1], gbs[2] + 0.15, f'G{b}', fontsize=7,
                    ha='center')

        ax.set_xlim(0, env.X - 1)
        ax.set_ylim(0, env.Y - 1)
        ax.set_zlim(0, env.Z - 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        title = subplot_titles[idx] if subplot_titles else f'Episode {idx+1}'
        ax.set_title(f'{title}\narr={arr_rate:.2f}  R={ep_reward:.2f}',
                     fontsize=10)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    else:
        matplotlib.use('TkAgg')
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Visualize multi-UAV trajectories')
    p.add_argument('--algo', choices=['qmix', 'iql', 'mappo', 'random'],
                   default='random')
    p.add_argument('--model_path', type=str, default=None,
                   help='Path to a single model checkpoint directory')
    p.add_argument('--checkpoints', nargs='+', default=None,
                   help='List of checkpoint directories (side-by-side comparison)')
    p.add_argument('--num_episodes', type=int, default=3,
                   help='Episodes to visualize (one per subplot when no checkpoints)')
    p.add_argument('--phi', type=float, default=None)
    p.add_argument('--rho_col', type=float, default=None)
    p.add_argument('--r_fail', type=float, default=None)
    p.add_argument('--v_max', type=int, default=None)
    p.add_argument('--T_max', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=None,
                   help='Output file path (pdf/png). If omitted, show interactively.')
    return p.parse_args()


def main():
    cli = parse_args()

    # Determine checkpoint list
    if cli.checkpoints:
        checkpoint_list = cli.checkpoints
        mode = 'compare'
    elif cli.model_path:
        checkpoint_list = [cli.model_path] * cli.num_episodes
        mode = 'episodes'
    else:
        checkpoint_list = [None] * cli.num_episodes
        mode = 'episodes'

    augment = cli.algo in ('qmix', 'iql')
    env = _build_env(cli, augment_obs=augment)

    # Build agent (once per unique checkpoint, or None for random)
    results = []
    subplot_titles = []

    if mode == 'compare':
        # One episode per checkpoint
        for ck_idx, ckpt in enumerate(checkpoint_list):
            seed = cli.seed + ck_idx
            label = os.path.basename(ckpt) if ckpt else 'random'
            if cli.algo == 'random' or ckpt is None:
                traj, arr, rew = _run_episode_random(env, seed=seed)
            elif cli.algo in ('qmix', 'iql'):
                agent, device = _load_qmix_agent(env, ckpt, cli.algo, cli)
                traj, arr, rew = _run_episode_qmix_iql(env, agent, seed=seed)
            elif cli.algo == 'mappo':
                agent, device = _load_mappo_agent(env, ckpt, cli)
                traj, arr, rew = _run_episode_mappo(env, agent, device, seed=seed)
            results.append((traj, arr, rew))
            subplot_titles.append(label)
    else:
        # Multiple episodes from the same checkpoint (or random)
        agent, device = None, None
        if cli.algo in ('qmix', 'iql') and checkpoint_list[0] is not None:
            agent, device = _load_qmix_agent(env, checkpoint_list[0], cli.algo, cli)
        elif cli.algo == 'mappo' and checkpoint_list[0] is not None:
            agent, device = _load_mappo_agent(env, checkpoint_list[0], cli)

        for ep in range(cli.num_episodes):
            seed = cli.seed + ep
            if cli.algo == 'random' or agent is None:
                traj, arr, rew = _run_episode_random(env, seed=seed)
            elif cli.algo in ('qmix', 'iql'):
                traj, arr, rew = _run_episode_qmix_iql(env, agent, seed=seed)
            elif cli.algo == 'mappo':
                traj, arr, rew = _run_episode_mappo(env, agent, device, seed=seed)
            results.append((traj, arr, rew))
            subplot_titles.append(f'Episode {ep + 1}')

    algo_label = cli.algo.upper()
    ckpt_label = (os.path.basename(cli.model_path)
                  if cli.model_path else 'random')
    fig_title = f'{algo_label} – {ckpt_label}  (○=start  ★=end  ▲=GBS)'

    if not cli.out:
        matplotlib.use('TkAgg')

    plot_trajectories(results, env, subplot_titles=subplot_titles,
                      fig_title=fig_title, out_path=cli.out)


if __name__ == '__main__':
    main()
