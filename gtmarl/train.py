"""Train graph-aided IPPO or a learning baseline.

Example
-------
python -m gtmarl.train --algo gippo --seed 0 --iters 1500 --out runs/main
python -m gtmarl.train --algo mappo --seed 0 --env N=30 K_b_scale_with_N=true
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch

from .config import EnvConfig, TrainConfig, apply_overrides, from_dict, to_dict
from .env import LAWNEnv
from .ppo import ALGO_NAMES, PPOTrainer, evaluate_model


def run_dir(tc: TrainConfig, ec: EnvConfig) -> str:
    name = f"{tc.algo}_N{ec.N}_M{ec.M}_s{tc.seed}" + (f"_{tc.tag}" if tc.tag else "")
    return os.path.join(tc.out, name)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="gippo", choices=ALGO_NAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--env", nargs="*", default=[], help="EnvConfig overrides key=value")
    p.add_argument("--train", nargs="*", default=[], help="TrainConfig overrides key=value")
    p.add_argument("--tb", action="store_true", help="also log to TensorBoard")
    p.add_argument("--skip_done", action="store_true", help="exit if this run already finished")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="continue RUN_DIR (its config.json) from model_last.pt up to --iters; --train "
                        "overrides apply on top; progress.csv is appended")
    a = p.parse_args(argv)

    if a.resume:
        with open(os.path.join(a.resume, "config.json")) as f:
            cfg0 = json.load(f)
        ec = apply_overrides(from_dict(EnvConfig, cfg0["env"]), a.env)
        tc = apply_overrides(from_dict(TrainConfig, cfg0["train"]), a.train)
        tc.init_from = ""
        a.algo, a.seed = tc.algo, tc.seed
    else:
        ec = apply_overrides(EnvConfig(), a.env)
        tc = apply_overrides(TrainConfig(), a.train)
    tc.algo, tc.seed = a.algo, a.seed
    for k in ("iters", "out", "tag", "device"):
        if getattr(a, k) is not None:
            setattr(tc, k, getattr(a, k))

    torch.set_num_threads(tc.threads)
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    out = a.resume.rstrip("/") if a.resume else run_dir(tc, ec)
    if a.skip_done and os.path.exists(os.path.join(out, "model_last.pt")):
        with open(os.path.join(out, "progress.csv")) as f:
            done = sum(1 for _ in f) - 1
        if done >= tc.iters:
            print(f"{out}: finished ({done} iterations), skipping", flush=True)
            return
    os.makedirs(out, exist_ok=True)
    start, t_prev, old_fields = 0, 0.0, None
    if a.resume and os.path.exists(os.path.join(out, "model_last.pt")):
        with open(os.path.join(out, "progress.csv")) as f:
            old = list(csv.DictReader(f))
        start, t_prev = len(old), float(old[-1]["time_min"]) if old else 0.0
        old_fields = list(old[0].keys()) if old else None
        print(f"{out}: resuming at iteration {start} -> {tc.iters}", flush=True)
    Ns = [int(x) for x in str(tc.N_set).replace(";", ",").split(",") if x] if tc.N_set else []
    if Ns:
        ec.state_N = max(Ns + [ec.N])            # one critic input size for every swarm size
    env = LAWNEnv(copy.deepcopy(ec), n_envs=tc.n_envs, seed=tc.seed)   # own config: curriculum below
    eval_env = LAWNEnv(ec, n_envs=tc.n_envs, seed=10_000 + tc.seed)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"env": to_dict(ec), "train": to_dict(tc), "radio": env.radio.summary(),
                   "K_eff": env.K, "resumed_at": start}, f, indent=2, default=str)

    trainer = PPOTrainer(tc.algo, env, tc)
    if start:
        trainer.model.load_state_dict(torch.load(os.path.join(out, "model_last.pt"), map_location=tc.device))
        trainer.it = start
    envs = [env]
    if Ns:                                       # swarm-size randomisation (one N per iteration)
        envs = [LAWNEnv(from_dict(EnvConfig, {**to_dict(ec), "N": n}),
                        n_envs=tc.n_envs, seed=tc.seed + 7 * k) for k, n in enumerate(Ns)]
        trainer.env = envs[0]
    writer = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out)
    log_f = open(os.path.join(out, "progress.csv"), "a" if start else "w", newline="")
    log_w = None
    if start and old_fields:
        log_w = csv.DictWriter(log_f, fieldnames=old_fields, extrasaction="ignore")
    t0 = time.time() - 60 * t_prev
    for it in range(start, tc.iters):
        trainer.env = envs[it % len(envs)]
        w_frac = min(1.0, it / tc.sep_warmup) if tc.sep_warmup > 0 else 1.0
        for e_ in envs:                          # separation curriculum (training envs only)
            e_.cfg.w_sep = ec.w_sep * w_frac
        row = trainer.train_iter()
        row["train/w_sep"] = trainer.env.cfg.w_sep
        row["iter"] = it
        row["time_min"] = (time.time() - t0) / 60
        if (it + 1) % tc.eval_every == 0 or it == tc.iters - 1:
            ev = evaluate_model(eval_env, trainer.model, trainer.device, tc.eval_episodes)
            row.update({f"eval/{k}": v for k, v in ev.items()})
            print(f"[{tc.algo} N={ec.N} s={tc.seed}] it {it + 1}/{tc.iters} "
                  f"succ {ev['success']:.3f} J {ev['J']:.3f} p_out {ev['p_out']:.4f} "
                  f"detour {ev['detour']:.3f} ({row['time_min']:.1f} min)", flush=True)
        if (it + 1) % tc.save_every == 0 or it == tc.iters - 1:
            torch.save(trainer.model.state_dict(), os.path.join(out, f"model_{it + 1}.pt"))
            torch.save(trainer.model.state_dict(), os.path.join(out, "model_last.pt"))
        if log_w is None:
            fields = sorted(set(row) | {f"eval/{k}" for k in
                                        ("success", "detour", "p_out", "jain_load", "social_cost", "J",
                                         "viol", "rate_ratio", "potential", "overload", "jain_util",
                                         "ep_len", "route_ok")})
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
