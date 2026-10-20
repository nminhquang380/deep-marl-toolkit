"""Empirical best-response (Nash) gap of a trained swarm (Remark 1).

UAV 0 deviates: its policy is re-trained by PPO on its own cost while all other
UAVs keep the frozen trained policy. Since missions are drawn i.i.d. and the
UAVs are homogeneous, UAV 0 is a representative player. The gap is

    gap = J_0(pi, pi_-0) - J_0(pi_BR, pi_-0) >= 0 (up to estimation error),

reported in absolute terms and relative to J_0(pi, pi_-0). A small gap means
no UAV can profit noticeably from a unilateral change of policy.

Example
-------
python -m gtmarl.br_gap --run runs/main/gippo_N20_M4_s0 --ckpt model_1500.pt --iters 300
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from .config import TrainConfig, apply_overrides
from .env import LAWNEnv
from .evaluate import load_run
from .models import ActorCritic
from .ppo import PPOTrainer, rollout


@torch.no_grad()
def agent0_cost(ec, learner, opponent, n_episodes, n_envs, seed):
    env = LAWNEnv(ec, n_envs=n_envs, seed=seed)
    costs, succ = [], []
    for _ in range(max(1, n_episodes // n_envs)):
        rollout(env, learner, torch.device("cpu"), opponent=opponent, learner_agents=np.array([0]))
        costs.append(env.stat_cost[:, 0].copy())
        succ.append(env.arrived[:, 0].copy())
    c = np.concatenate(costs)
    return float(c.mean()), float(c.std() / np.sqrt(len(c))), float(np.concatenate(succ).mean())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="model_last.pt")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--eval_episodes", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", nargs="*", default=[], help="TrainConfig overrides for the deviator")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)

    ec, tc0, policy = load_run(a.run, a.ckpt)
    # deviator: individual critic (it optimises its own J_0), warm-started from the trained actor
    learner = ActorCritic(ec.M, policy.actor[-1].out_features, tc0.hidden, tc0.emb_dim,
                          nbr=policy.nbr, knn_k=policy.knn_k, state_dim=0)
    learner.encoder.load_state_dict(policy.encoder.state_dict())
    learner.actor.load_state_dict(policy.actor.state_dict())
    if not policy.central:
        learner.critic.load_state_dict(policy.critic.state_dict())
        learner.vnorm.load_state_dict(policy.vnorm.state_dict())
    opponent = copy.deepcopy(policy).eval()

    # J_0 when UAV 0 follows the trained policy
    j_pi, se_pi, s_pi = agent0_cost(ec, policy, opponent, a.eval_episodes, a.n_envs, seed=777)

    tc = apply_overrides(TrainConfig(), a.train)
    tc.iters, tc.n_envs, tc.seed = a.iters, a.n_envs, a.seed
    tc.ent_coef = tc.ent_coef_final               # fine-tuning, little extra exploration
    env = LAWNEnv(ec, n_envs=tc.n_envs, seed=a.seed + 100)
    algo = {"graph": "gippo", "knn": "ippo_mlp", "none": "ippo"}[policy.nbr]
    tr = PPOTrainer(algo, env, tc, model=learner,
                    opponent=opponent, learner_agents=np.array([0]))
    curve = []
    for it in range(a.iters):
        tr.train_iter()
        if (it + 1) % max(1, a.iters // 6) == 0:
            j_br, se_br, _ = agent0_cost(ec, tr.model, opponent, 512, a.n_envs, seed=777)
            curve.append((it + 1, j_br))
            print(f"BR it {it + 1}: J0 {j_br:.4f} (trained policy {j_pi:.4f})", flush=True)

    j_br, se_br, s_br = agent0_cost(ec, tr.model, opponent, a.eval_episodes, a.n_envs, seed=777)
    res = {"run": a.run, "ckpt": a.ckpt, "algo": tc0.algo, "N": ec.N,
           "J0_policy": j_pi, "J0_policy_se": se_pi, "succ0_policy": s_pi,
           "J0_br": j_br, "J0_br_se": se_br, "succ0_br": s_br,
           "gap": j_pi - j_br, "gap_rel": (j_pi - j_br) / max(abs(j_pi), 1e-9), "curve": curve}
    print(json.dumps(res, indent=1))
    out = a.out or os.path.join(a.run, f"brgap_{os.path.splitext(a.ckpt)[0]}_s{a.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
