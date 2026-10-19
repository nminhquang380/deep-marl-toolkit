"""Train graph-aided IPPO or a learning baseline.

Example
-------
python -m gtmarl.train --algo gippo --seed 0 --iters 1500 --out runs/main
python -m gtmarl.train --algo mappo --seed 0 --env N=30 K_b_scale_with_N=true
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from .config import EnvConfig, TrainConfig, apply_overrides, to_dict
from .env import LAWNEnv
from .ppo import PPOTrainer, evaluate_model


def run_dir(tc: TrainConfig, ec: EnvConfig) -> str:
    name = f"{tc.algo}_N{ec.N}_M{ec.M}_s{tc.seed}" + (f"_{tc.tag}" if tc.tag else "")
    return os.path.join(tc.out, name)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="gippo", choices=["gippo", "ippo", "mappo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--env", nargs="*", default=[], help="EnvConfig overrides key=value")
    p.add_argument("--train", nargs="*", default=[], help="TrainConfig overrides key=value")
    p.add_argument("--tb", action="store_true", help="also log to TensorBoard")
    a = p.parse_args(argv)

    ec = apply_overrides(EnvConfig(), a.env)
    tc = apply_overrides(TrainConfig(), a.train)
    tc.algo, tc.seed = a.algo, a.seed
    for k in ("iters", "out", "tag", "device"):
        if getattr(a, k) is not None:
            setattr(tc, k, getattr(a, k))

    torch.set_num_threads(tc.threads)
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    out = run_dir(tc, ec)
    os.makedirs(out, exist_ok=True)
    env = LAWNEnv(ec, n_envs=tc.n_envs, seed=tc.seed)
    eval_env = LAWNEnv(ec, n_envs=tc.n_envs, seed=10_000 + tc.seed)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"env": to_dict(ec), "train": to_dict(tc), "radio": env.radio.summary(),
                   "K_eff": env.K}, f, indent=2, default=str)

    trainer = PPOTrainer(tc.algo, env, tc)
    writer = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out)
    log_f = open(os.path.join(out, "progress.csv"), "w", newline="")
    log_w = None
    t0 = time.time()
    for it in range(tc.iters):
        row = trainer.train_iter()
        row["iter"] = it
        row["time_min"] = (time.time() - t0) / 60
        if (it + 1) % tc.eval_every == 0 or it == tc.iters - 1:
            ev = evaluate_model(eval_env, trainer.model, trainer.device, tc.eval_episodes)
            row.update({f"eval/{k}": v for k, v in ev.items()})
            print(f"[{tc.algo} N={ec.N} s={tc.seed}] it {it + 1}/{tc.iters} "
                  f"succ {ev['success']:.3f} J {ev['J']:.3f} viol {ev['viol']:.4f} "
                  f"detour {ev['detour']:.3f} ({row['time_min']:.1f} min)", flush=True)
        if (it + 1) % tc.save_every == 0 or it == tc.iters - 1:
            torch.save(trainer.model.state_dict(), os.path.join(out, f"model_{it + 1}.pt"))
            torch.save(trainer.model.state_dict(), os.path.join(out, "model_last.pt"))
        if log_w is None:
            fields = sorted(set(row) | {f"eval/{k}" for k in
                                        ("success", "detour", "J", "welfare", "viol", "rate_ratio",
                                         "jain_util", "jain_load", "overload", "ep_len")})
            log_w = csv.DictWriter(log_f, fieldnames=fields, extrasaction="ignore")
            log_w.writeheader()
        log_w.writerow(row)
        log_f.flush()
        if writer is not None:
            for k, v in row.items():
                if k != "iter":
                    writer.add_scalar(k, v, it)
    log_f.close()


if __name__ == "__main__":
    main()
