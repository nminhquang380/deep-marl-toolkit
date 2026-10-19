"""Graph-aided IPPO (Alg. 1) and the MAPPO baseline.

* gippo : shared actor-critic on o_i, own reward, own value  (proposed)
* ippo  : as gippo without the neighbour summary              (ablation)
* mappo : shared actor on o_i, centralised critic on the joint state,
          team reward = mean of the individual rewards        (cooperative benchmark)

No discount is applied (gamma = 1), matching Eq. (20) and the shaping
invariance of Eq. (24). An agent's trajectory ends at its last active slot,
after which its value is zero (Eq. (32)).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .env import LAWNEnv
from .models import ActorCritic


def build_model(algo: str, env: LAWNEnv, tc: TrainConfig) -> ActorCritic:
    return ActorCritic(
        n_gbs=env.M, n_actions=env.A, hidden=tc.hidden, emb_dim=tc.emb_dim,
        use_graph=(algo != "ippo"), state_dim=env.state_dim if algo == "mappo" else 0,
    )


def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


@torch.no_grad()
def rollout(env: LAWNEnv, model: ActorCritic, device, opponent: Optional[ActorCritic] = None,
            learner_agents: Optional[np.ndarray] = None, deterministic: bool = False) -> Dict:
    """Run E episodes to completion.

    If ``opponent`` is given, agents not in ``learner_agents`` act with the frozen
    opponent policy (used for the best-response gap).
    """
    o = env.reset()
    E, N = env.E, env.N
    buf = {k: [] for k in ("obs", "state", "mask", "act", "logp", "val", "rew", "active")}
    lmask = np.ones(N, bool)
    if opponent is not None:
        lmask = np.zeros(N, bool)
        lmask[learner_agents] = True
    while not env.done.all():
        obs_t, mask_t = _t(o["obs"], device), _t(o["mask"], device, torch.bool)
        d = model.dist(obs_t, mask_t)
        a = d.probs.argmax(-1) if deterministic else d.sample()
        if opponent is not None:
            d_op = opponent.dist(obs_t, mask_t)
            a_op = d_op.probs.argmax(-1) if deterministic else d_op.sample()
            a = torch.where(_t(lmask, device, torch.bool)[None], a, a_op)
        logp = d.log_prob(a)
        state_t = _t(o["state"], device)
        v = model.vnorm.denormalize(model.value(obs_t, state_t))
        nxt = env.step(a.cpu().numpy())
        buf["obs"].append(o["obs"]); buf["state"].append(o["state"]); buf["mask"].append(o["mask"])
        buf["act"].append(a.cpu().numpy()); buf["logp"].append(logp.cpu().numpy())
        buf["val"].append(v.cpu().numpy()); buf["rew"].append(nxt["reward"])
        buf["active"].append(nxt["active_before"])
        o = nxt
    out = {k: np.stack(v) for k, v in buf.items()}
    out["learner"] = lmask
    out["metrics"] = env.episode_metrics()
    return out


def gae_individual(rew, val, active, lam):
    """rew, val, active: (T, E, N). Returns advantages and returns."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros_like(rew[0])
    for t in reversed(range(T)):
        if t + 1 < T:
            nonterm = active[t + 1].astype(np.float32)
            v_next = val[t + 1]
        else:
            nonterm = np.zeros_like(rew[0]); v_next = np.zeros_like(rew[0])
        delta = rew[t] + nonterm * v_next - val[t]
        last = delta + lam * nonterm * last
        adv[t] = last * active[t]
    return adv, (adv + val) * active


def gae_team(rew_team, val, env_active, lam):
    """rew_team, val, env_active: (T, E)."""
    return gae_individual(rew_team[..., None], val[..., None], env_active[..., None], lam)


class PPOTrainer:
    def __init__(self, algo: str, env: LAWNEnv, tc: TrainConfig, model: Optional[ActorCritic] = None,
                 opponent: Optional[ActorCritic] = None, learner_agents: Optional[np.ndarray] = None):
        self.algo, self.env, self.tc = algo, env, tc
        self.device = torch.device(tc.device)
        self.model = (model or build_model(algo, env, tc)).to(self.device)
        self.opponent = opponent.to(self.device).eval() if opponent is not None else None
        self.learner_agents = learner_agents
        self.opt = torch.optim.Adam(self.model.parameters(), lr=tc.lr, eps=1e-5)
        self.it = 0

    def collect(self) -> Dict:
        return rollout(self.env, self.model, self.device, self.opponent, self.learner_agents)

    def update(self, ro: Dict, frac: float) -> Dict[str, float]:
        tc, dev = self.tc, self.device
        T, E, N = ro["rew"].shape
        active = ro["active"] & ro["learner"][None, None, :]
        if self.algo == "mappo":
            env_act = ro["active"].any(-1)
            rew_team = ro["rew"].sum(-1) / N
            adv_e, ret_e = gae_team(rew_team, ro["val"], env_act, tc.gae_lambda)
            adv = np.broadcast_to(adv_e, (T, E, N))
            v_idx = np.argwhere(env_act)
            v_state = ro["state"][v_idx[:, 0], v_idx[:, 1]]
            v_ret = ret_e[v_idx[:, 0], v_idx[:, 1], 0]
        else:
            adv, ret = gae_individual(ro["rew"], ro["val"], ro["active"], tc.gae_lambda)

        idx = np.argwhere(active)
        sel = (idx[:, 0], idx[:, 1], idx[:, 2])
        b_obs = _t(ro["obs"][sel], dev)
        b_mask = _t(ro["mask"][sel], dev, torch.bool)
        b_act = _t(ro["act"][sel], dev, torch.long)
        b_logp = _t(ro["logp"][sel], dev)
        b_adv = adv[sel]
        b_adv = _t((b_adv - b_adv.mean()) / (b_adv.std() + 1e-8), dev)
        if self.algo == "mappo":
            c_state, c_ret = _t(v_state, dev), _t(v_ret, dev)
        else:
            c_ret = _t(ret[sel], dev)
        self.model.vnorm.update(c_ret)
        c_ret = self.model.vnorm.normalize(c_ret)

        ent_coef = tc.ent_coef + frac * (tc.ent_coef_final - tc.ent_coef)
        n = len(idx)
        mb = max(1, n // tc.minibatches)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        n_upd = 0
        for _ in range(tc.epochs):
            perm = torch.randperm(n, device=dev)
            if self.algo == "mappo":
                nv = len(c_ret)
                vperm = torch.randperm(nv, device=dev)
                vmb = max(1, nv // tc.minibatches)
            for k in range(tc.minibatches):
                j = perm[k * mb:(k + 1) * mb]
                d = self.model.dist(b_obs[j], b_mask[j])
                logp = d.log_prob(b_act[j])
                ratio = torch.exp(logp - b_logp[j])
                a_j = b_adv[j]
                pg = -torch.min(ratio * a_j, ratio.clamp(1 - tc.clip, 1 + tc.clip) * a_j).mean()
                ent = d.entropy().mean()
                if self.algo == "mappo":
                    jv = vperm[k * vmb:(k + 1) * vmb]
                    v = self.model.value(None, c_state[jv])
                    v_loss = ((v - c_ret[jv]) ** 2).mean()
                else:
                    v = self.model.value(b_obs[j])
                    v_loss = ((v - c_ret[j]) ** 2).mean()
                loss = pg - ent_coef * ent + tc.vf_coef * v_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pg_loss"] += pg.item(); stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > tc.clip).float().mean().item()
                n_upd += 1
        stats = {k: v / n_upd for k, v in stats.items()}
        stats["samples"] = n
        return stats

    def train_iter(self) -> Dict[str, float]:
        frac = min(1.0, self.it / max(1, self.tc.iters - 1))
        ro = self.collect()
        st = self.update(ro, frac)
        self.it += 1
        rew = ro["rew"][:, :, ro["learner"]].sum(0).mean()
        out = {f"train/{k}": v for k, v in ro["metrics"].items()}
        out.update({f"loss/{k}": v for k, v in st.items()})
        out["train/shaped_return"] = float(rew)
        return out


@torch.no_grad()
def evaluate_model(env: LAWNEnv, model: ActorCritic, device, n_episodes: int,
                   deterministic: bool = False) -> Dict[str, float]:
    return _avg([rollout(env, model, device, deterministic=deterministic)["metrics"]
                 for _ in range(max(1, n_episodes // env.E))])


def _avg(ms: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ms[0].keys()
    return {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
