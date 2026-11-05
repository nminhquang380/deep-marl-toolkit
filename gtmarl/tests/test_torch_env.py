"""Equivalence of the torch environment with the NumPy reference (same missions, same actions)."""
import numpy as np
import pytest
import torch

from gtmarl.config import EnvConfig
from gtmarl.env import LAWNEnv
from gtmarl.torch_env import TorchLAWNEnv


def _random_admissible(mask, rng):
    p = mask / mask.sum(-1, keepdims=True)
    return (rng.random(mask.shape[:2] + (1,)) > p.cumsum(-1)).sum(-1)


def _knn_sorted(x, env):
    """kNN block as a multiset per UAV: equidistant neighbours may come in a different order."""
    o0 = env.own_dim + env.nbr_dim + 1
    blk = x[..., o0:o0 + env.nbr_dim * env.cap].reshape(*x.shape[:-1], env.cap, env.nbr_dim)
    keys = np.round(blk, 4)
    flat = keys.reshape(-1, env.cap, env.nbr_dim)
    return np.stack([flat[i][np.lexsort(flat[i].T[::-1])] for i in range(flat.shape[0])])


CASES = [
    {},                                                     # TVT model (default)
    {"N": 20, "penalty": "lost", "eta": 1.0},               # other penalty, failures
    {"N": 12, "sep_mode": "mask", "exit_inactive": False, "d_safe": 1.0, "sep_inclusive": False},
    {"N": 15, "ablate": ("vel", "cocell"), "obs_version": 2},
]


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kw", CASES)
def test_equivalence(kw, dev):
    cfg = EnvConfig(**kw)
    a = LAWNEnv(cfg, n_envs=3, seed=11)
    b = TorchLAWNEnv(cfg, n_envs=3, seed=11, device=dev)
    oa, ob = a.reset(), b.reset()
    rng = np.random.default_rng(0)
    o0 = a.own_dim + a.nbr_dim + 1
    kb = slice(o0, o0 + a.nbr_dim * a.cap)
    steps = 0
    while not a.done.all():
        ob = {k: v.cpu() for k, v in ob.items()}
        assert np.array_equal(oa["mask"], ob["mask"].numpy())
        xa, xb = oa["obs"], ob["obs"].numpy()
        rest = np.ones(xa.shape[-1], bool); rest[kb] = False
        np.testing.assert_allclose(xa[..., rest], xb[..., rest], atol=1e-5)
        # kNN block: same multiset of neighbours when all fit (|M_i| <= cap); with more, the cut may pick
        # different equidistant neighbours, so compare the selected distances
        ka, kb_ = _knn_sorted(xa, a), _knn_sorted(xb, a)
        cnt = np.round(xa[..., a.own_dim + a.nbr_dim] * 8).reshape(-1)
        fit = cnt <= a.cap
        np.testing.assert_allclose(ka[fit], kb_[fit], atol=1e-4)
        np.testing.assert_allclose(np.sort(ka[~fit][..., 3], -1), np.sort(kb_[~fit][..., 3], -1), atol=1e-4)
        np.testing.assert_allclose(oa["state"], ob["state"].numpy(), atol=1e-5)
        act = _random_admissible(oa["mask"], rng)
        oa, ob = a.step(act), b.step(torch.as_tensor(act, device=dev))
        for k in ("reward", "cost", "n_sep", "penalty"):
            np.testing.assert_allclose(oa[k], ob[k].cpu().numpy(), atol=1e-4, err_msg=k)
        assert np.array_equal(a.done, b.done.cpu().numpy())
        steps += 1
    assert b.done.all() and steps > 5
    ma, mb = a.episode_metrics(), b.episode_metrics()
    for k, v in ma.items():
        if np.isfinite(v):
            assert abs(v - mb[k]) < 1e-6 * max(1.0, abs(v)), (k, v, mb[k])


@pytest.mark.parametrize("dev", DEVICES)
def test_torch_rollout_and_train(dev):
    from gtmarl.config import TrainConfig
    from gtmarl.ppo import PPOTrainer
    for algo in ("egat_cc", "mappo_egat", "ippo_mlp", "egat_aux"):
        e = TorchLAWNEnv(EnvConfig(N=10), n_envs=2, seed=1, device=dev)
        tr = PPOTrainer(algo, e, TrainConfig(n_envs=2, epochs=1, minibatches=2, device=dev))
        st = tr.train_iter()
        assert np.isfinite(st["loss/v_loss"]) and "train/success" in st
