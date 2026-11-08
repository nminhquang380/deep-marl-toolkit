"""One test episode per seed for SP, EGAT-IPPO and IPPO-MLP (same missions), saved with per-episode metrics."""
import sys, json
import numpy as np, torch
from gtmarl.evaluate import load_run, save_traj, heuristic_traj
from gtmarl.env import LAWNEnv
from gtmarl.ppo import rollout

RES, OUT = sys.argv[1], sys.argv[2]
runs = {"egat": f"{RES}/A/main/egat_cc_N50_M4_s0", "ippo": f"{RES}/A/main/ippo_mlp_N50_M4_s0"}
torch.manual_seed(0)
summary = {}
for seed in range(1000, 1012):
    row = {}
    for name, run in runs.items():
        ec, tc, model = load_run(run)
        env = LAWNEnv(ec, n_envs=1, seed=seed)
        torch.manual_seed(seed)
        rollout(env, model, torch.device("cpu"), deterministic=False, with_value=False)
        save_traj(env, f"{OUT}/traj_{name}_{seed}.npz")
        m = env.episode_metrics(); row[name] = {k: m[k] for k in ("success", "J", "p_out", "detour", "los", "overload")}
    heuristic_traj(ec, "greedy", seed, f"{OUT}/traj_sp_{seed}.npz")
    env = LAWNEnv(ec, n_envs=1, seed=seed)       # metrics of SP on the same missions
    from gtmarl.heuristics import run_heuristic
    m = run_heuristic(env, "greedy", 1, seed=seed); row["sp"] = {k: m[k] for k in ("success", "J", "p_out", "detour", "los", "overload")}
    summary[seed] = row
    print(seed, {k: (round(v["J"], 1), round(v["success"], 2), round(v["p_out"], 4)) for k, v in row.items()}, flush=True)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
