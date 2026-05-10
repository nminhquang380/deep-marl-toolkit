import argparse
import os
import sys
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.arguments import get_common_args
from marltoolkit.envs.multi_uav.multi_uav_env import MultiUAVEnv
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def run_random_episode(env):
    obs, state, _ = env.reset()
    done = False
    episode_reward = 0.0
    episode_step = 0

    while not done:
        avail = env.get_available_actions()  # (N, n_actions)
        # Sample uniformly from available actions for each agent
        actions = np.array([
            np.random.choice(np.where(avail[i])[0])
            for i in range(env.num_agents)
        ])
        _, _, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        episode_reward += reward
        episode_step += 1

    return {
        'episode_reward': episode_reward,
        'episode_step': episode_step,
        'win_rate': env.win_counted,
    }


def main():
    common_args = get_common_args()
    args = common_args
    args.algo_name = 'random'

    env = MultiUAVEnv(args=args)

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

    steps_cnt = 0
    episode_cnt = 0
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        res = run_random_episode(env)
        episode_cnt += 1
        steps_cnt += res['episode_step']

        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Random] episode: {}, step: {}, win_rate: {:.2f}, reward: {:.4f}'
                .format(episode_cnt, steps_cnt, res['win_rate'],
                        res['episode_reward']))
            logger.log_train_data(res, steps_cnt)

        progress_bar.update(res['episode_step'])


if __name__ == '__main__':
    main()
