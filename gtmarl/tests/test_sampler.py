"""The occupancy-grid episode samplers draw exactly the episodes of the original full-grid samplers."""
import types

import numpy as np
import pytest

from gtmarl.config import EnvConfig
from gtmarl.env import LAWNEnv


def _origins_ref(self):
    N, V, sep = self.N, self._all_vox, self.sep_init
    n_pool = len(self._origin_pool)
    counts = np.full(n_pool, N // n_pool)
    counts[self.rng.permutation(n_pool)[: N % n_pool]] += 1
    orig = []
    for pool, c in zip(self._origin_pool, counts):
        got = 0
        if c == 0:
            continue
        for idx in self.rng.permutation(pool):
            v = V[idx]
            if not orig or np.min(np.linalg.norm(np.asarray(orig) - v, axis=1)) >= sep:
                orig.append(v)
                got += 1
                if got == c:
                    break
        if got < c:
            return None
    return np.asarray(orig)[self.rng.permutation(N)]


def _dests_ref(self, orig):
    cfg, N, V, sep = self.cfg, self.N, self._all_vox, self.sep_init
    free = np.min(np.linalg.norm(V[:, None, :] - orig[None], axis=-1), axis=1) >= sep
    dest = []
    for i in range(N):
        idx = np.flatnonzero(free & (np.abs(V - orig[i]).sum(1) >= cfg.min_od_l1))
        if len(idx) == 0:
            return None
        v = V[self.rng.choice(idx)]
        dest.append(v)
        free &= np.linalg.norm(V - v, axis=1) >= sep
    return np.asarray(dest)


CASES = [
    {},                                                                     # TVT model, N = 50
    {"N": 60},
    {"N": 20, "sep_mode": "mask", "exit_inactive": False, "d_safe": 1.0, "sep_inclusive": False},
    {"N": 30, "d_safe": 2.0},
    {"N": 25, "origin_mode": "uniform"},
]


@pytest.mark.parametrize("kw", CASES)
def test_sampler_identical(kw):
    cfg = EnvConfig(**kw)
    a = LAWNEnv(cfg, n_envs=6, seed=5)
    b = LAWNEnv(cfg, n_envs=6, seed=5)
    b._sample_origins_once = types.MethodType(_origins_ref, b)
    b._sample_dests = types.MethodType(_dests_ref, b)
    for _ in range(2):
        a.reset(), b.reset()
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.dest, b.dest)
        assert np.array_equal(a.route_ok, b.route_ok)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
