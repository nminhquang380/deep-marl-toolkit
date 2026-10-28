"""Non-learning baselines operating on the same observations / admissible sets.

* greedy : shortest-path progress toward the destination, congestion-unaware
* myopic : one-step best response to the current cell demands, i.e. it minimises
           the estimated time-to-go plus the congestion penalty of Eq. (12) (and the
           separation cost) at the landing voxel with the other UAVs held at their
           current positions

Both escape deadlocks under the separation mask of Eq. (17): a UAV that has
hovered for ``patience`` consecutive slots takes a random admissible move.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .env import LAWNEnv


def _choose(score: np.ndarray, mask: np.ndarray, rng) -> np.ndarray:
    score = np.where(mask, score + 1e-6 * rng.random(score.shape), np.inf)
    return score.argmin(-1)


def greedy_actions(env: LAWNEnv, mask: np.ndarray, rng) -> np.ndarray:
    land = env.pos[:, :, None, :] + env.acts[None, None]
    togo = np.abs(land - env.dest[:, :, None, :]).sum(-1).astype(float)
    return _choose(togo, mask, rng)


def myopic_actions(env: LAWNEnv, mask: np.ndarray, rng) -> np.ndarray:
    cfg = env.cfg
    land = env.pos[:, :, None, :] + env.acts[None, None]
    land_c = np.clip(land, 0, env.dims - 1)
    idx = tuple(np.moveaxis(land_c, -1, 0))
    b_land = env.radio.serve[idx]                                    # (E,N,A)
    m_land = env.radio.m[idx]
    D_land = np.take_along_axis(env.D[:, None, :].repeat(env.N, 1), b_land, axis=2)
    own_here = (b_land == env.b[..., None]) * env.m_cur[..., None]
    D_new = D_land - own_here + m_land
    g_hat = np.minimum(m_land, np.maximum(D_new - env.K, 0))
    togo = np.abs(land - env.dest[:, :, None, :]).sum(-1) / cfg.v_max
    score = (1.0 + togo) / env.Tmin[..., None] + cfg.mu * g_hat
    if cfg.sep_mode != "mask":                       # separation cost with the others held in place
        d = np.linalg.norm(land[:, :, :, None, :] - env.pos[:, None, None, :, :], axis=-1)   # (E,N,A,N)
        near = (d < cfg.d_safe - 1e-9) & env.present[:, None, None, :]
        near[:, np.arange(env.N), :, np.arange(env.N)] = False
        score = score + cfg.w_sep * near.sum(-1)
    return _choose(score, mask, rng)


POLICIES = {"greedy": greedy_actions, "myopic": myopic_actions}


def escape_deadlock(a: np.ndarray, mask: np.ndarray, stuck: np.ndarray, patience: int, rng) -> np.ndarray:
    moves = mask.copy()
    moves[..., 0] = False
    can = moves.any(-1) & (stuck >= patience)
    if can.any():
        p = moves / np.maximum(moves.sum(-1, keepdims=True), 1)
        rnd = (rng.random(stuck.shape + (1,)) > p.cumsum(-1)).sum(-1)
        a = np.where(can, np.minimum(rnd, mask.shape[-1] - 1), a)
    return a


def run_heuristic(env: LAWNEnv, name: str, n_episodes: int, seed: int = 0,
                  patience: int = 2) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    fn = POLICIES[name]
    ms: List[Dict[str, float]] = []
    for _ in range(max(1, n_episodes // env.E)):
        o = env.reset()
        stuck = np.zeros((env.E, env.N), np.int64)
        while not env.done.all():
            a = escape_deadlock(fn(env, o["mask"], rng), o["mask"], stuck, patience, rng)
            stuck = np.where((a == 0) & env.active, stuck + 1, 0)
            o = env.step(a)
        ms.append(env.episode_metrics())
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
