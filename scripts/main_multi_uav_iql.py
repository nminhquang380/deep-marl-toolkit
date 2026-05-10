import argparse
import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.arguments import get_common_args
from configs.multi_uav_iql_config import MultiUAVIQLConfig
from marltoolkit.agents.qmix_agent import QMixAgent
from marltoolkit.data import ReplayBuffer
from marltoolkit.envs.multi_uav.multi_uav_env import MultiUAVEnv
from marltoolkit.modules.actors import RNNActorModel
from marltoolkit.runners.episode_runner import (run_eval_episode,
                                                run_train_episode)
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def main():
    iql_config = MultiUAVIQLConfig()
    common_args = get_common_args()
    args = argparse.Namespace(**vars(iql_config))
    args.__dict__.update(**vars(common_args))

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda
                          else 'cpu')

    # Environment
    env = MultiUAVEnv(args=args)
    args.episode_limit = env.episode_limit
    args.obs_dim = env.obs_dim
    args.obs_shape = env.obs_shape
    args.state_dim = env.state_dim
    args.state_shape = env.state_shape
    args.num_agents = env.num_agents
    args.n_actions = env.n_actions
    args.action_shape = env.action_shape
    args.reward_shape = env.reward_shape
    args.done_shape = env.done_shape
    args.device = device
    args.num_eval_episodes = args.num_eval_episodes or args.num_test_episodes

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

    # Actor input shape includes optional last-actions / agent-id augmentation
    args.obs_shape = env.get_actor_input_shape()

    rpm = ReplayBuffer(
        max_size=args.replay_buffer_size,
        num_agents=args.num_agents,
        episode_limit=args.episode_limit,
        obs_shape=args.obs_shape,
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        reward_shape=args.reward_shape,
        done_shape=args.done_shape,
        device=device,
    )

    actor_model = RNNActorModel(args)

    agent = QMixAgent(
        actor_model=actor_model,
        mixer_model=None,
        num_agents=args.num_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        egreedy_exploration=args.egreedy_exploration,
        min_exploration=args.min_exploration,
        target_update_tau=args.target_update_tau,
        target_update_interval=args.target_update_interval,
        learner_update_freq=args.learner_update_freq,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
    )

    progress_bar = ProgressBar(args.memory_warmup_size)
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(env, agent, rpm, args)
        progress_bar.update()

    steps_cnt = 0
    episode_cnt = 0
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        train_res_dict = run_train_episode(env, agent, rpm, args)
        episode_cnt += 1
        steps_cnt += train_res_dict['episode_step']

        agent.learning_rate = max(
            agent.lr_scheduler.step(train_res_dict['episode_step']),
            agent.min_learning_rate,
        )
        train_res_dict.update({
            'exploration': agent.exploration,
            'learning_rate': agent.learning_rate,
            'replay_max_size': rpm.size(),
            'target_update_count': agent.target_update_count,
        })

        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Train] episode: {}, step: {}, win_rate: {:.2f}, reward: {:.4f}'
                .format(episode_cnt, steps_cnt, train_res_dict['win_rate'],
                        train_res_dict['episode_reward']))
            logger.log_train_data(train_res_dict, steps_cnt)

        if episode_cnt % args.test_log_interval == 0:
            eval_res_dict = run_eval_episode(env, agent, args=args)
            text_logger.info(
                '[Eval]  episode: {}, step: {:.1f}, win_rate: {:.2f}, reward: {:.4f}'
                .format(episode_cnt, eval_res_dict['episode_step'],
                        eval_res_dict['win_rate'],
                        eval_res_dict['episode_reward']))
            logger.log_test_data(eval_res_dict, steps_cnt)

        if episode_cnt % args.save_interval == 0:
            model_dir = os.path.join(log_path, 'models', f'ep_{episode_cnt:07d}')
            agent.save_model(save_dir=model_dir)

        progress_bar.update(train_res_dict['episode_step'])


if __name__ == '__main__':
    main()
