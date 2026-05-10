import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.arguments import get_common_args
from configs.multi_uav_mappo_config import MultiUAVMAPPOConfig
from marltoolkit.agents.mappo_agent import MAPPOAgent
from marltoolkit.envs.multi_uav.multi_uav_env import MultiUAVEnv
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def collect_episode(env, agent, args, device, deterministic=False):
    """Collect one full episode and return transitions + stats."""
    agent.prep_rollout()
    obs, state, _ = env.reset()

    obs_list, state_list, actions_list, log_probs_list = [], [], [], []
    values_list, rewards_list, dones_list, avail_list = [], [], [], []

    actor_rnn = None
    critic_rnn = None
    done = False
    episode_reward = 0.0
    episode_step = 0

    while not done:
        avail = env.get_available_actions()  # (N, n_actions)

        obs_t = torch.FloatTensor(obs).to(device)          # (N, obs_dim)
        # Centralized critic: repeat global state for all agents
        state_t = torch.FloatTensor(state).unsqueeze(0).expand(
            args.num_agents, -1).contiguous().to(device)  # (N, state_dim)
        masks_t = torch.ones(args.num_agents, 1).to(device)
        avail_t = torch.FloatTensor(avail).to(device)

        with torch.no_grad():
            values, actions, log_probs, actor_rnn, critic_rnn = agent.get_actions(
                obs_t, state_t, masks_t, avail_t,
                actor_rnn, critic_rnn, deterministic)

        actions_np = actions.cpu().numpy().flatten()  # (N,)
        next_obs, next_state, reward, terminated, truncated, info = env.step(
            actions_np)
        done = terminated or truncated

        obs_list.append(obs)
        state_list.append(state)
        actions_list.append(actions.cpu().numpy())       # (N, 1)
        log_probs_list.append(log_probs.cpu().numpy())   # (N, 1)
        values_list.append(values.cpu().numpy())         # (N, 1)
        rewards_list.append(reward)
        dones_list.append(float(done))
        avail_list.append(avail)

        obs, state = next_obs, next_state
        episode_reward += reward
        episode_step += 1

    # Bootstrap value at episode end
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).expand(
            args.num_agents, -1).contiguous().to(device)
        masks_t = torch.zeros(args.num_agents, 1).to(device)
        bootstrap_val = agent.get_values(state_t, masks_t, critic_rnn)
    values_list.append(bootstrap_val.cpu().numpy())  # (N, 1)

    T = episode_step
    N = args.num_agents

    # Stack arrays: (T, N, ...)
    obs_arr = np.stack(obs_list, axis=0)            # (T, N, obs_dim)
    state_arr = np.stack(state_list, axis=0)        # (T, state_dim)
    actions_arr = np.stack(actions_list, axis=0)    # (T, N, 1)
    log_probs_arr = np.stack(log_probs_list, axis=0)  # (T, N, 1)
    values_arr = np.stack(values_list, axis=0)      # (T+1, N, 1)
    rewards_arr = np.array(rewards_list)            # (T,) scalar
    dones_arr = np.array(dones_list)                # (T,)
    avail_arr = np.stack(avail_list, axis=0)        # (T, N, n_actions)

    # GAE returns
    returns = np.zeros((T, N, 1), dtype=np.float32)
    gae = np.zeros((N, 1), dtype=np.float32)
    for t in reversed(range(T)):
        r = rewards_arr[t]
        v = values_arr[t]        # (N, 1)
        v_next = values_arr[t + 1]  # (N, 1)
        mask = 1.0 - dones_arr[t]
        delta = r + args.gamma * mask * v_next - v
        gae = delta + args.gamma * args.gae_lambda * mask * gae
        returns[t] = gae + v

    advantages = returns - values_arr[:T]  # (T, N, 1)

    # Repeat state for each agent: (T, N, state_dim)
    state_exp = np.expand_dims(state_arr, axis=1).repeat(N, axis=1)

    # Flatten to (T*N, ...)
    obs_flat = obs_arr.reshape(T * N, -1)
    state_flat = state_exp.reshape(T * N, -1)
    actions_flat = actions_arr.reshape(T * N, 1)
    log_probs_flat = log_probs_arr.reshape(T * N, 1)
    values_flat = values_arr[:T].reshape(T * N, 1)
    returns_flat = returns.reshape(T * N, 1)
    adv_flat = advantages.reshape(T * N, 1)
    avail_flat = avail_arr.reshape(T * N, args.n_actions)
    # masks: 1 everywhere except last step per episode (simplified: all 1)
    masks_flat = np.ones((T * N, 1), dtype=np.float32)
    active_flat = np.ones((T * N, 1), dtype=np.float32)

    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-5)

    # PPO updates
    agent.prep_training()
    n_samples = T * N
    mini_batch = max(1, n_samples // args.num_mini_batch)
    train_info = {'policy_loss': 0.0, 'value_loss': 0.0, 'dist_entropy': 0.0}
    n_updates = 0

    for _ in range(args.ppo_epoch):
        perm = np.random.permutation(n_samples)
        for start in range(0, n_samples, mini_batch):
            idx = perm[start:start + mini_batch]
            batch = (
                torch.FloatTensor(obs_flat[idx]).to(device),
                torch.FloatTensor(state_flat[idx]).to(device),
                torch.LongTensor(actions_flat[idx]).to(device),
                torch.FloatTensor(values_flat[idx]).to(device),
                torch.FloatTensor(returns_flat[idx]).to(device),
                torch.FloatTensor(masks_flat[idx]).to(device),
                torch.FloatTensor(active_flat[idx]).to(device),
                torch.FloatTensor(log_probs_flat[idx]).to(device),
                torch.FloatTensor(adv_flat[idx]).to(device),
                torch.FloatTensor(avail_flat[idx]).to(device),
                None,  # actor_rnn_states (non-recurrent)
                None,  # critic_rnn_states (non-recurrent)
            )
            p_loss, v_loss, entropy, _, _, _ = agent.ppo_update(batch)
            train_info['policy_loss'] += p_loss.item()
            train_info['value_loss'] += v_loss.item()
            train_info['dist_entropy'] += entropy.item()
            n_updates += 1

    if n_updates > 0:
        for k in train_info:
            train_info[k] /= n_updates

    train_info['episode_reward'] = episode_reward
    train_info['episode_step'] = episode_step
    train_info['win_rate'] = env.win_counted
    return train_info


def eval_policy(env, agent, args, device, num_eval=5):
    rewards, steps, wins = [], [], []
    for _ in range(num_eval):
        ep = collect_episode(env, agent, args, device, deterministic=True)
        rewards.append(ep['episode_reward'])
        steps.append(ep['episode_step'])
        wins.append(ep['win_rate'])
    return {
        'episode_reward': float(np.mean(rewards)),
        'episode_step': float(np.mean(steps)),
        'win_rate': float(np.mean(wins)),
    }


def main():
    mappo_config = MultiUAVMAPPOConfig()
    common_args = get_common_args()
    # Merge: common_args as base, mappo_config overrides (so MAPPO booleans win)
    args = argparse.Namespace(**vars(common_args))
    args.__dict__.update(**vars(mappo_config))

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda
                          else 'cpu')
    args.device = device

    # Environment
    env = MultiUAVEnv(args=args)
    args.episode_limit = env.episode_limit
    args.obs_dim = env.obs_dim
    args.state_dim = env.state_dim
    args.state_shape = env.state_shape
    args.num_agents = env.num_agents
    args.n_actions = env.n_actions
    args.num_eval_episodes = args.num_eval_episodes or args.num_test_episodes

    # MAPPO-specific: actor sees augmented obs (last actions + agent id)
    args.actor_input_dim = env.get_actor_input_shape()[0]
    # Discrete action space object required by ACTLayer
    args.action_space = gym.spaces.Discrete(args.n_actions)

    # Logger
    scenario = 'multi_uav'
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = os.path.join(args.project, scenario, args.algo_name,
                            timestamp).replace(os.path.sep, '_')
    log_path = os.path.join(args.log_dir, args.project, scenario, args.algo_name)
    tensorboard_log_path = get_outdir(log_path, 'tensorboard_log_dir')
    log_file = os.path.join(log_path, log_name + '.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')

    if args.logger == 'wandb':
        logger = WandbLogger(
            train_interval=args.train_log_interval,
            test_interval=args.test_log_interval,
            update_interval=args.train_log_interval,
            project=args.project,
            name=log_name,
            save_interval=1,
            config=args,
        )
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    if args.logger == 'tensorboard':
        logger = TensorboardLogger(writer)
    else:
        logger.load(writer)

    agent = MAPPOAgent(args)

    steps_cnt = 0
    episode_cnt = 0
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        train_res = collect_episode(env, agent, args, device)
        episode_cnt += 1
        steps_cnt += train_res['episode_step']

        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Train] episode: {}, step: {}, win_rate: {:.2f}, reward: {:.4f}'
                .format(episode_cnt, steps_cnt, train_res['win_rate'],
                        train_res['episode_reward']))
            logger.log_train_data(train_res, steps_cnt)

        if episode_cnt % args.test_log_interval == 0:
            eval_res = eval_policy(env, agent, args, device,
                                   num_eval=args.num_eval_episodes)
            text_logger.info(
                '[Eval]  episode: {}, step: {:.1f}, win_rate: {:.2f}, reward: {:.4f}'
                .format(episode_cnt, eval_res['episode_step'],
                        eval_res['win_rate'], eval_res['episode_reward']))
            logger.log_test_data(eval_res, steps_cnt)

        if episode_cnt % args.save_interval == 0:
            model_dir = os.path.join(log_path, 'models', f'ep_{episode_cnt:07d}')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(agent.actor_model.state_dict(),
                       os.path.join(model_dir, 'actor_model.pt'))
            torch.save(agent.critic_model.state_dict(),
                       os.path.join(model_dir, 'critic_model.pt'))

        progress_bar.update(train_res['episode_step'])


if __name__ == '__main__':
    main()
