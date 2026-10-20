"""Sanity checks of the environment against the manuscript.

Run:  python -m pytest -q gtmarl/tests
"""
import numpy as np
import pytest

from gtmarl.config import EnvConfig
from gtmarl.env import LAWNEnv


def _random_admissible(mask, rng):
    E, N, A = mask.shape
    p = mask / mask.sum(-1, keepdims=True)
    c = p.cumsum(-1)
    u = rng.random((E, N, 1))
    return (u > c).sum(-1)


@pytest.fixture
def env():
    return LAWNEnv(EnvConfig(N=30, K_b=25, init_check="off"), n_envs=4, seed=1)  # congested


def test_coverage():
    e = LAWNEnv(EnvConfig(), n_envs=1, seed=0)
    assert e.radio.covered, e.radio.summary()


def test_penalty_certifies_rate(env):
    """Eq. (13): g_i = 0  <=>  D_b <= K_b  <=>  R_i >= R_min."""
    rng = np.random.default_rng(0)
    o = env.reset()
    for _ in range(30):
        o = env.step(_random_admissible(o["mask"], rng))
        g = env.penalty()
        Db = np.take_along_axis(env.D, env.b, 1)
        assert np.array_equal(g == 0, Db <= env.K)


def test_separation_preserved(env):
    """Eq. (17) keeps Eq. (9) from a separated initial configuration."""
    rng = np.random.default_rng(1)
    for _ in range(3):
        o = env.reset()
        while not env.done.all():
            o = env.step(_random_admissible(o["mask"], rng), check_mask=True)
            d = np.linalg.norm(env.pos[:, :, None] - env.pos[:, None], axis=-1)
            d[:, np.arange(env.N), np.arange(env.N)] = np.inf
            assert d.min() >= env.cfg.d_safe - 1e-9


def test_exact_potential(env):
    """Prop. 1: unilateral deviation changes c_i and phi by the same amount."""
    rng = np.random.default_rng(2)
    o = env.reset()
    for _ in range(10):
        o = env.step(_random_admissible(o["mask"], rng))
    mask = o["mask"]
    base = _random_admissible(mask, rng)
    saved = (env.pos.copy(), env.t.copy(), env.arrived.copy(), env.failed.copy())
    checked = 0
    for e in range(env.E):
        for i in range(env.N):
            if not env.active[e, i]:
                continue
            alts = np.flatnonzero(mask[e, i])
            for a2 in alts[:3]:
                vals = []
                for a in (base, None):
                    acts = base.copy()
                    if a is None:
                        acts[e, i] = a2
                    env.pos, env.t, env.arrived, env.failed = (x.copy() for x in saved)
                    act_before = env.active.copy()
                    out = env.step(acts)
                    c_i = 1.0 / env.Tmin[e, i] + env.cfg.mu * out["penalty"][e, i]
                    vals.append((c_i, env.potential()[e]))
                    assert act_before[e, i]
                (c1, p1), (c2, p2) = vals
                assert abs((c1 - c2) - (p1 - p2)) < 1e-9
                checked += 1
    assert checked > 0


def test_shaping_telescopes():
    """Eq. (24): sum_t r_nav = rho_nav * ||q_i^0 - d_i|| for every UAV."""
    e = LAWNEnv(EnvConfig(N=10), n_envs=3, seed=3)
    rng = np.random.default_rng(3)
    o = e.reset()
    d0 = np.linalg.norm(e.pos - e.dest, axis=-1)
    tot_r = np.zeros((e.E, e.N)); tot_c = np.zeros((e.E, e.N))
    while not e.done.all():
        o = e.step(_random_admissible(o["mask"], rng))
        tot_r += o["reward"]; tot_c += o["cost"]
    nav = tot_r + tot_c
    assert np.allclose(nav, e.cfg.w_nav * d0, atol=1e-4)


def test_awgn_runs():
    e = LAWNEnv(EnvConfig(N=10, fading="none"), n_envs=2, seed=4)
    rng = np.random.default_rng(4)
    o = e.reset()
    while not e.done.all():
        o = e.step(_random_admissible(o["mask"], rng))
    assert np.isfinite(e.episode_metrics()["J"])
