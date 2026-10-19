"""Evaluate trained runs and heuristic baselines, optionally across swarm sizes.

Examples
--------
# trained runs at their own N, plus N = 10..50 without retraining (gippo / ippo)
python -m gtmarl.evaluate --runs runs/main/gippo_N20_M4_s* --N 10 20 30 40 50 \
       --env K_b_scale_with_N=true --episodes 512 --out results/scal_gippo.csv

# heuristics
python -m gtmarl.evaluate --heuristics greedy myopic --N 10 20 30 40 50 \
       --env K_b_scale_with_N=true --out results/scal_heur.csv

# dump one episode of trajectories for plotting
python -m gtmarl.evaluate --runs runs/main/gippo_N20_M4_s0 --traj results/traj_gippo.npz
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np
import torch

from .config import EnvConfig, TrainConfig, apply_overrides, from_dict
from .env import LAWNEnv
from .heuristics import run_heuristic
from .ppo import build_model, evaluate_model, rollout


def load_run(run: str, ckpt: str = "model_last.pt", device: str = "cpu"):
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    ec = from_dict(EnvConfig, cfg["env"])
    tc = from_dict(TrainConfig, cfg["train"])
    env = LAWNEnv(ec, n_envs=1)
    model = build_model(tc.algo, env, tc)
    model.load_state_dict(torch.load(os.path.join(run, ckpt), map_location=device))
    model.eval()
    return ec, tc, model


def save_traj(env: LAWNEnv, path: str):
    traj = np.stack(env.traj)                      # (T+1, E, N, 3)
    np.savez_compressed(path, traj=traj[:, 0], dest=env.dest[0], orig=env.orig[0],
                        Tmin=env.Tmin[0], H=env.Hdl[0], T_arr=env.T_arr[0], arrived=env.arrived[0],
                        gbs_vox=env.radio.gbs_vox, serve=env.radio.serve, level=env.radio.level,
                        dims=env.dims, K=env.K)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--ckpt", default="model_last.pt")
    p.add_argument("--heuristics", nargs="*", default=[])
    p.add_argument("--N", nargs="*", type=int, default=None, help="swarm sizes (default: trained N)")
    p.add_argument("--env", nargs="*", default=[], help="EnvConfig overrides applied on top")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--traj", default=None, help="save one episode of trajectories (first run)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(1)

    runs: List[str] = []
    for r in a.runs:
        runs += sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r]
    rows: List[Dict] = []

    for run in runs:
        ec0, tc, model = load_run(run, a.ckpt)
        for N in (a.N or [ec0.N]):
            if tc.algo == "mappo" and N != ec0.N:
                continue                           # centralised critic is tied to N (actor is not,
                                                   # but MAPPO is retrained per N for fairness)
            ec = apply_overrides(from_dict(EnvConfig, vars(ec0)), a.env)
            ec.N = N
            env = LAWNEnv(ec, n_envs=a.n_envs, seed=a.seed)
            if a.traj:
                env1 = LAWNEnv(ec, n_envs=1, seed=a.seed)
                rollout(env1, model, torch.device("cpu"), deterministic=a.deterministic)
                save_traj(env1, a.traj)
                print("saved", a.traj)
                a.traj = None
            m = evaluate_model(env, model, torch.device("cpu"), a.episodes, a.deterministic)
            row = {"method": tc.algo, "seed": tc.seed, "N_train": ec0.N, "N": N, "M": ec.M,
                   "K": env.K, "run": run, **m}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "run"},
                  flush=True)

    for h in a.heuristics:
        for N in (a.N or [EnvConfig().N]):
            ec = apply_overrides(EnvConfig(), a.env)
            ec.N = N
            env = LAWNEnv(ec, n_envs=a.n_envs, seed=a.seed)
            m = run_heuristic(env, h, a.episodes, seed=a.seed)
            row = {"method": h, "seed": 0, "N_train": 0, "N": N, "M": ec.M, "K": env.K, "run": "", **m}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "run"},
                  flush=True)

    if a.out and rows:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
