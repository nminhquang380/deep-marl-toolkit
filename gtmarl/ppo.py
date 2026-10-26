"""Graph-aided IPPO (Alg. 1) and the benchmarks.

Every learner shares the actor pi(a_i | o_i) across UAVs. They differ in the neighbour
encoder, in what the critic observes, and in the reward the critic and advantage use:

* own reward  (r_i)          -> each UAV solves its own best-response problem (non-cooperative)
* team reward (mean_j r_j)   -> social cost (cooperative, CTDE-MAPPO)
* local critic V(o_i) | agent-specific centralised critic V_i(s, o_i): an action-independent
  baseline, so it changes the variance of the policy gradient but not the objective.

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


ALGOS = {  # algo -> (neighbour encoder, centralised critic, team reward, congestion aux head)
    "egat": ("egat", False, False, False),        # proposed: EGAT encoder, own reward, local critic
    "egat_aux": ("egat", False, False, True),     #   + congestion-prediction head
    "egat_cc": ("egat", True, False, False),      #   + agent-specific centralised critic V_i(s, o_i)
    "egat_cc_aux": ("egat", True, False, True),
    "gippo": ("graph", False, False, False),      # mean of raw beacon features (v22 Eq. (29))
    "gippo_msg": ("graph_msg", False, False, False),  # message MLP before the mean (GraphSAGE-mean)
    "ippo_mlp": ("knn", False, False, False),     # IPPO-MLP benchmark (k nearest neighbours)
    "ippo": ("none", False, False, False),        # no neighbour information
    "mappo_egat": ("egat", True, True, False),    # CTDE-MAPPO benchmark: team reward + central critic
    "mappo": ("graph", True, True, False),        # CTDE-MAPPO with the mean encoder (legacy runs)
}
ALGO_NAMES = list(ALGOS)


def spec(algo: str) -> Dict[str, object]:
    nbr, central, team, aux = ALGOS[algo]
    return {"nbr": nbr, "central": central, "team": team, "aux": aux}


def build_model(algo: str, env: LAWNEnv, tc: TrainConfig) -> ActorCritic:
    sp = spec(algo)
    central = sp["central"]
    model = ActorCritic(n_gbs=env.M, n_actions=env.A, hidden=tc.hidden, emb_dim=tc.emb_dim,
                        nbr=sp["nbr"], knn_k=env.k, state_dim=env.state_dim if central else 0,
                        obs_dim=env.obs_dim if central else 0, nbr_cap=env.cap,
                        own_dim=env.own_dim, nbr_dim=env.nbr_dim, aux=sp["aux"],
                        heads=getattr(tc, "heads", 4), gnn_layers=getattr(tc, "gnn_layers", 2))
    if getattr(tc, "init_from", ""):
        # warm start (e.g. CTDE-MAPPO from a trained IPPO policy): copy encoder, message MLP and
        # actor; every such tensor must exist with the same shape, otherwise fail loudly
        sd = torch.load(tc.init_from, map_location="cpu")
        own = model.state_dict()
        keys = [k for k in own if k.startswith(("encoder.", "actor.", "phi.", "gnn."))]
        bad = [k for k in keys if k not in sd or sd[k].shape != own[k].shape]
        if bad:
            raise ValueError(f"init_from {tc.init_from}: incompatible tensors {bad[:4]} "
                             "(source and target must use the same neighbour encoder)")
        own.update({k: sd[k] for k in keys})
        model.load_state_dict(own)
    return model


def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


@torch.no_grad()
def rollout(env: LAWNEnv, model: ActorCritic, device, opponent: Optional[ActorCritic] = None,
            learner_agents: Optional[np.ndarray] = None, deterministic: bool = False,
            with_value: bool = True) -> Dict:
    """Run E episodes to completion (with_value=False for evaluation: actor only).

    If ``opponent`` is given, agents not in ``learner_agents`` act with the frozen
    opponent policy (used for the best-response gap).
    """
    o = env.reset()
    E, N = env.E, env.N
    buf = {k: [] for k in ("obs", "state", "mask", "act", "logp", "val", "rew", "active", "dem_next")}
    lmask = np.ones(N, bool)
    if opponent is not None:
        lmask = np.zeros(N, bool)
        lmask[learner_agents] = True
    while not env.done.all():
        # only active UAVs act (an inactive UAV can only hover, action 0), so the networks run on them alone
        act_np = env.active
        rows = np.nonzero(act_np.reshape(-1))[0]
        flat = lambda x: x.reshape(E * N, *x.shape[2:])
        obs_t = _t(flat(o["obs"])[rows], device)
        mask_t = _t(flat(o["mask"])[rows], device, torch.bool)
        state_t = _t(o["state"][rows // N], device)
        d, v_n, _ = model.forward_all(obs_t, mask_t, state_t, with_value=with_value)
        a_r = d.probs.argmax(-1) if deterministic else d.sample()
        if opponent is not None:
            d_op, _, _ = opponent.forward_all(obs_t, mask_t, state_t, with_value=False)
            a_op = d_op.probs.argmax(-1) if deterministic else d_op.sample()
            a_r = torch.where(_t(lmask[rows % N], device, torch.bool), a_r, a_op)
        logp_r = d.log_prob(a_r)
        a_full = np.zeros(E * N, np.int64); a_full[rows] = a_r.cpu().numpy()
        lp_full = np.zeros(E * N, np.float32); lp_full[rows] = logp_r.cpu().numpy()
        v_full = np.zeros(E * N, np.float32); v_full[rows] = model.vnorm.denormalize(v_n).cpu().numpy()
        if model.central and with_value:   # team critics need V(s, o_i) of every UAV of a live episode (cheap: no encoder)
            rv = np.nonzero(np.repeat(~env.done, N))[0]
            v_full[rv] = model.vnorm.denormalize(model.value(_t(flat(o["obs"])[rv], device),
                                                             _t(o["state"][rv // N], device))).cpu().numpy()
        a = torch.as_tensor(a_full.reshape(E, N))
        logp = torch.as_tensor(lp_full.reshape(E, N))
        v = torch.as_tensor(v_full.reshape(E, N))
        nxt = env.step(a.cpu().numpy())
        buf["obs"].append(o["obs"]); buf["state"].append(o["state"]); buf["mask"].append(o["mask"])
        buf["act"].append(a.cpu().numpy()); buf["logp"].append(logp.cpu().numpy())
        buf["val"].append(v.cpu().numpy()); buf["rew"].append(nxt["reward"])
        buf["active"].append(nxt["active_before"])
        buf["dem_next"].append((env.D / env.K).astype(np.float32))       # D_b(t+1) / K_b, (E, M)
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
                 opponent: Optional[ActorCritic] = None, learner_agents: Optional[np.ndarray] = None,
                 sp: Optional[Dict] = None):
        self.algo, self.env, self.tc = algo, env, tc
        self.sp = sp or spec(algo)
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
        team, central = self.sp["team"], self.sp["central"]
        use_aux = self.sp["aux"] and getattr(self.model, "aux", False)
        if team:
            # team reward (social cost / N) for every agent; per-agent value V(s, o_i);
            # every agent's trajectory lasts while any UAV of its episode is active
            crit_act = np.broadcast_to(ro["active"].any(-1, keepdims=True), (T, E, N))
            rew = np.broadcast_to(ro["rew"].sum(-1, keepdims=True) / N, (T, E, N))
        else:
            crit_act, rew = active, ro["rew"]
        adv, ret = gae_individual(rew, ro["val"], crit_act if team else ro["active"], tc.gae_lambda)
        v_idx = np.argwhere(crit_act)
        vsel = (v_idx[:, 0], v_idx[:, 1], v_idx[:, 2])

        idx = np.argwhere(active)
        sel = (idx[:, 0], idx[:, 1], idx[:, 2])
        b_obs = _t(ro["obs"][sel], dev)
        b_mask = _t(ro["mask"][sel], dev, torch.bool)
        b_act = _t(ro["act"][sel], dev, torch.long)
        b_logp = _t(ro["logp"][sel], dev)
        b_adv = adv[sel]
        b_adv = _t((b_adv - b_adv.mean()) / (b_adv.std() + 1e-8), dev)
        c_obs = _t(ro["obs"][vsel], dev) if team else b_obs
        c_state = _t(ro["state"][vsel[0], vsel[1]], dev) if central else None
        c_ret = _t(ret[vsel], dev)
        if use_aux:
            a_tgt = _t(ro["dem_next"][sel[0], sel[1]], dev)
        self.model.vnorm.update(c_ret)
        c_ret = self.model.vnorm.normalize(c_ret)
        actor_on = self.it >= getattr(tc, "critic_warmup", 0)

        ent_coef = tc.ent_coef + frac * (tc.ent_coef_final - tc.ent_coef)
        n, nv = len(idx), len(c_ret)
        mb, vmb = max(1, n // tc.minibatches), max(1, nv // tc.minibatches)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "aux_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                 "clipfrac": 0.0}
        n_upd = 0
        for _ in range(tc.epochs):
            perm = torch.randperm(n, device=dev)
            vperm = torch.randperm(nv, device=dev)
            for k in range(tc.minibatches):
                j = perm[k * mb:(k + 1) * mb]
                jv = vperm[k * vmb:(k + 1) * vmb]
                if team:                                   # critic samples differ from actor samples
                    d = self.model.dist(b_obs[j], b_mask[j])
                    v = self.model.value(c_obs[jv], c_state[jv])
                    z = self.model._z(b_obs[j]) if use_aux else None
                    v_tgt = c_ret[jv]
                else:                                      # same samples: one encoder pass
                    d, v, z = self.model.forward_all(b_obs[j], b_mask[j], c_state[j] if central else None)
                    v_tgt = c_ret[j]
                logp = d.log_prob(b_act[j])
                ratio = torch.exp(logp - b_logp[j])
                a_j = b_adv[j]
                pg = -torch.min(ratio * a_j, ratio.clamp(1 - tc.clip, 1 + tc.clip) * a_j).mean()
                ent = d.entropy().mean()
                v_loss = ((v - v_tgt) ** 2).mean()
                loss = tc.vf_coef * v_loss
                if actor_on:
                    loss = loss + pg - ent_coef * ent
                aux_l = torch.zeros(())
                if use_aux:
                    aux_l = ((self.model.aux_head(z) - a_tgt[j]) ** 2).mean()
                    loss = loss + tc.aux_coef * aux_l
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pg_loss"] += pg.item(); stats["v_loss"] += v_loss.item()
                    stats["aux_loss"] += aux_l.item()
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
    return _avg([rollout(env, model, device, deterministic=deterministic, with_value=False)["metrics"]
                 for _ in range(max(1, n_episodes // env.E))])


def _avg(ms: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ms[0].keys()
    return {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
