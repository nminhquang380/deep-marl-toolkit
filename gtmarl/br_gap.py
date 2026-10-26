"""Empirical best-response (Nash) gap, Eq. (36) of manuscript v22.

UAV 0 deviates: starting from the trained policy, its own policy is re-trained by PPO
on its own cost while all other UAVs keep the frozen trained policy. UAV indices are
shuffled every episode, so UAV 0 is a uniformly random player of the homogeneous swarm;
running several seeds gives the deviator set I of Eq. (36).

Estimator
---------
* Candidates: the trained policy itself (iteration 0) and snapshots of the deviator
  every ``iters / n_snap`` iterations. The approximate best response pi_hat is the
  candidate with the lowest J_0 on a validation set of missions, so it can never be
  worse than not deviating except through test noise (a fine-tuned deviator can get
  worse, e.g. when a rare missed deadline costs c_fail).
* Test: J_0 of pi and of pi_hat on the same test missions (common random numbers;
  missions depend only on the environment seed), giving a paired estimate of
  gap = J_0(pi) - J_0(pi_hat) with its standard error.

Example
-------
python -m gtmarl.br_gap --run gtmarl/data/gtmarl_runs/main/gippo_N30_M4_s0 --ckpt model_1500.pt
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

VAL_SEED, TEST_SEED = 778, 777


@torch.no_grad()
def agent0_costs(ec, learner, opponent, n_episodes, n_envs, seed):
    """Per-episode cost and arrival of UAV 0 (same missions for the same seed)."""
    env = LAWNEnv(ec, n_envs=n_envs, seed=seed)
    costs, succ = [], []
    for _ in range(max(1, n_episodes // n_envs)):
        rollout(env, learner, torch.device("cpu"), opponent=opponent, learner_agents=np.array([0]),
                with_value=False)
        costs.append(env.stat_cost[:, 0].copy())
        succ.append(env.arrived[:, 0].copy())
    return np.concatenate(costs), np.concatenate(succ)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="model_last.pt")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--n_snap", type=int, default=6)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--val_episodes", type=int, default=1024)
    p.add_argument("--eval_episodes", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", nargs="*", default=[], help="TrainConfig overrides for the deviator")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    ec, tc0, policy = load_run(a.run, a.ckpt)
    # deviator: own critic (it minimises its own J_0), warm-started from the trained actor
    kw = dict(policy.kw, state_dim=0, obs_dim=0, aux=False)
    learner = ActorCritic(**kw)
    own_sd = learner.state_dict()
    src = policy.state_dict()
    shared = [k for k in own_sd if k.startswith(("encoder.", "actor.", "phi.", "gnn."))]
    if not policy.central:
        shared += [k for k in own_sd if k.startswith(("critic.", "vnorm."))]
    own_sd.update({k: src[k] for k in shared})
    learner.load_state_dict(own_sd)
    opponent = copy.deepcopy(policy).eval()

    # fine-tuning defaults: one agent's samples per iteration -> smaller steps
    tc = TrainConfig(lr=1e-4, epochs=4, minibatches=2)
    tc = apply_overrides(tc, a.train)
    tc.iters, tc.n_envs, tc.seed = a.iters, a.n_envs, a.seed
    tc.ent_coef = tc.ent_coef_final
    env = LAWNEnv(ec, n_envs=tc.n_envs, seed=a.seed + 100)
    sp = {"nbr": policy.nbr, "central": False, "team": False, "aux": False}   # own reward, own critic
    tr = PPOTrainer(tc0.algo, env, tc, model=learner, opponent=opponent, learner_agents=np.array([0]),
                    sp=sp)

    # candidate 0 = the trained policy (no deviation)
    c0, _ = agent0_costs(ec, policy, opponent, a.val_episodes, a.n_envs, VAL_SEED)
    best_val, best_it, best_state = float(c0.mean()), 0, None
    curve = [(0, best_val)]
    every = max(1, a.iters // a.n_snap)
    for it in range(a.iters):
        tr.train_iter()
        if (it + 1) % every == 0 or it == a.iters - 1:
            c, _ = agent0_costs(ec, tr.model, opponent, a.val_episodes, a.n_envs, VAL_SEED)
            curve.append((it + 1, float(c.mean())))
            print(f"BR it {it + 1}: val J0 {c.mean():.4f} (policy {curve[0][1]:.4f})", flush=True)
            if c.mean() < best_val:
                best_val, best_it, best_state = float(c.mean()), it + 1, copy.deepcopy(tr.model.state_dict())

    # paired test on fresh missions
    c_pi, s_pi = agent0_costs(ec, policy, opponent, a.eval_episodes, a.n_envs, TEST_SEED)
    if best_state is None:
        c_br, s_br = c_pi, s_pi
    else:
        learner.load_state_dict(best_state)
        c_br, s_br = agent0_costs(ec, learner, opponent, a.eval_episodes, a.n_envs, TEST_SEED)
    d = c_pi - c_br
    j_pi = float(c_pi.mean())
    res = {"run": a.run, "ckpt": a.ckpt, "algo": tc0.algo, "N": ec.N, "seed": a.seed,
           "J0_policy": j_pi, "succ0_policy": float(s_pi.mean()),
           "J0_br": float(c_br.mean()), "succ0_br": float(s_br.mean()),
           "gap": float(d.mean()), "gap_se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "gap_rel": float(d.mean()) / max(abs(j_pi), 1e-9),
           "best_iter": best_it, "val_curve": curve}
    print(json.dumps(res, indent=1))
    out = a.out or os.path.join(a.run, f"brgap_{os.path.splitext(a.ckpt)[0]}_s{a.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
