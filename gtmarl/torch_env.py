"""GPU (PyTorch) implementation of the LAWN Markov game, numerically equivalent to env.LAWNEnv.

All E episodes and N UAVs are advanced with tensor operations on one device, so a rollout keeps
the environment, the policy and the buffers on the GPU and only copies the finished batch back
(see ppo.rollout_torch). Episodes (origins, destinations) are still sampled by the NumPy helper,
once per reset, since the rejection sampling of Sec. II-A is not worth vectorising.

Equivalence with LAWNEnv is checked in tests/test_torch_env.py: identical missions and actions
give identical costs, rewards, observations (up to the order of equidistant neighbours, which
the permutation-invariant encoders do not see) and metrics.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .config import EnvConfig
from .env import CDF_GRID, LAWNEnv

F32, F64 = torch.float32, torch.float64


class TorchLAWNEnv:
    """E parallel episodes of G on a torch device (same interface as LAWNEnv for the learners)."""

    is_torch = True

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: Optional[int] = None, device="cpu"):
        h = LAWNEnv(cfg, n_envs=1, seed=seed)       # radio map, constants and episode sampler
        self.helper, self.cfg, self.radio = h, cfg, h.radio
        self.dev = torch.device(device)
        self.E, self.N, self.M, self.K, self.A = n_envs, h.N, h.M, h.K, h.A
        for k in ("k", "cap", "v2", "own_dim", "nbr_dim", "obs_dim", "state_N", "state_dim",
                  "mask_mode", "sep_move"):
            setattr(self, k, getattr(h, k))
        d = self.dev
        self.acts = torch.as_tensor(h.acts, device=d)                            # (A, 3) long
        self.dims = torch.as_tensor(h.dims, device=d)
        self.dims_norm = torch.as_tensor(h.dims_norm, dtype=F32, device=d)
        r = h.radio
        self._RH = (cfg.R * cfg.H, cfg.H)
        self.serve = torch.as_tensor(r.serve.reshape(-1), device=d)              # voxel -> cell
        self.mvox = torch.as_tensor(r.m.reshape(-1), dtype=F64, device=d)        # voxel -> bearer
        self.rhovox = torch.as_tensor(r.rho_vox.reshape(-1), dtype=F64, device=d)
        self.gbs_vox = torch.as_tensor(r.gbs_vox, dtype=F32, device=d)
        self.m_max = float(r.m_max)
        self.eye = torch.eye(self.N, dtype=torch.bool, device=d)

    # ------------------------------------------------------------------ reset
    def reset(self) -> Dict[str, torch.Tensor]:
        E, N, d, cfg = self.E, self.N, self.dev, self.cfg
        pos, dest, ok = np.zeros((E, N, 3), np.int64), np.zeros((E, N, 3), np.int64), np.zeros(E, bool)
        for e in range(E):
            pos[e], dest[e], ok[e] = self.helper._sample_episode()
        self.pos = torch.as_tensor(pos, device=d)
        self.dest = torch.as_tensor(dest, device=d)
        self.route_ok = ok
        self.orig = self.pos.clone()
        self.prev = self.pos.clone()
        self.Tmin = torch.ceil((self.pos - self.dest).abs().sum(-1).to(F64) / cfg.v_max)
        self.Hdl = torch.floor(cfg.eta * self.Tmin + 1e-9)
        self.c_fail_i = cfg.c_fail + (cfg.c_fail_mode == "lexi") * cfg.mu * self.m_max * self.Hdl
        self.t = torch.zeros(E, dtype=F64, device=d)
        self.arrived = torch.zeros(E, N, dtype=torch.bool, device=d)
        self.failed = torch.zeros_like(self.arrived)
        self.T_arr = torch.full((E, N), -1.0, dtype=F64, device=d)
        self._radio_update()
        z = lambda *s: torch.zeros(*s, dtype=F64, device=d)
        self.stat_cost, self.stat_viol, self.stat_out = z(E, N), z(E, N), z(E, N)
        self.stat_active_slots, self.stat_rate_ratio = z(E, N), z(E, N)
        self.stat_los, self.stat_nmac = z(E, N), z(E, N)
        self.stat_load_jain, self.stat_overload, self.stat_potential, self.stat_slots = z(E), z(E), z(E), z(E)
        self.stat_nbr = z(2)
        return self._obs()

    # -------------------------------------------------------------- dynamics
    @property
    def active(self) -> torch.Tensor:
        return ~(self.arrived | self.failed)

    @property
    def present(self) -> torch.Tensor:
        return self.active if self.cfg.exit_inactive else torch.ones_like(self.arrived)

    @property
    def done(self) -> torch.Tensor:
        return ~self.active.any(1)

    def _vidx(self, pos):
        return pos[..., 0] * self._RH[0] + pos[..., 1] * self._RH[1] + pos[..., 2]

    def _radio_update(self):
        idx = self._vidx(self.pos)
        self.b = self.serve[idx]
        self.m_cur = self.mvox[idx]
        self.rho_cur = self.rhovox[idx]
        self.D = torch.zeros(self.E, self.M, dtype=F64, device=self.dev).scatter_add_(
            1, self.b, self.m_cur * self.present)

    def served_fraction(self) -> torch.Tensor:
        s = torch.clamp(self.K / self.D.clamp_min(1e-9), max=1.0)
        return s.gather(1, self.b)

    def penalty(self) -> torch.Tensor:
        pen = self.cfg.penalty
        if pen == "lost":
            return self.m_cur * (1.0 - self.served_fraction())
        if pen == "shortfall":
            return 1.0 - self.served_fraction()
        if pen == "indicator":
            return (self.D.gather(1, self.b) > self.K).to(F64)
        over = torch.clamp(self.D - self.K, min=0.0)
        return torch.minimum(self.m_cur, over.gather(1, self.b)) * self.present

    def too_close(self, d: torch.Tensor) -> torch.Tensor:
        return d <= self.cfg.d_safe + 1e-9 if self.cfg.sep_inclusive else d < self.cfg.d_safe - 1e-9

    def _pdist(self, a, b=None):
        b = a if b is None else b
        return (a.unsqueeze(-2) - b.unsqueeze(-3)).to(F64).norm(dim=-1)

    def separation(self) -> torch.Tensor:
        pr = self.present
        return self.too_close(self._pdist(self.pos)) & pr[:, :, None] & pr[:, None, :] & ~self.eye

    def potential(self) -> torch.Tensor:
        phi = self.cfg.mu * torch.clamp(self.D - self.K, min=0.0).sum(1)
        if not self.mask_mode:
            phi = phi + self.cfg.w_sep * self.separation().sum((1, 2)).to(F64) / 2
        return phi

    def action_mask(self) -> torch.Tensor:
        land = self.pos[:, :, None, :] + self.acts[None, None]                   # (E,N,A,3)
        mask = ((land >= 0) & (land < self.dims)).all(-1)
        if self.mask_mode:
            cur = self._pdist(self.pos)
            nbr = (cur <= self.cfg.d_comm) & ~self.eye
            dist = (land[:, :, :, None, :] - self.pos[:, None, None, :, :]).to(F64).norm(dim=-1)
            dist = torch.where(nbr[:, :, None, :], dist, torch.full_like(dist, float("inf")))
            mask = mask & (dist.min(-1).values >= self.sep_move)
        mask[..., 0] = True
        inact = ~self.active
        mask[inact] = False
        mask[..., 0] |= inact
        return mask

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        actions = torch.as_tensor(actions, device=self.dev).long()
        act = self.active.clone()
        actf = act.to(F64)
        live = (~self.done).to(F64)
        t_old = self.t.clone()
        dist_old = (self.pos - self.dest).to(F64).norm(dim=-1)
        self.prev = self.pos.clone()
        self.pos = self.pos + self.acts[actions] * act[..., None]
        self.t = self.t + live
        arrived_now = act & (self.pos == self.dest).all(-1)
        failing = act & ~arrived_now & (self.t[:, None] >= self.Hdl)
        self.T_arr = torch.where(arrived_now, self.t[:, None].expand_as(self.T_arr), self.T_arr)
        self.arrived |= arrived_now
        self.failed |= failing
        self._radio_update()

        g = self.penalty()
        cost = actf * (1.0 / self.Tmin + cfg.mu * g)                             # Eq. (21), third branch
        n_sep = torch.zeros_like(g)
        if not self.mask_mode:
            n_sep = self.separation().sum(-1).to(F64)
            cost = cost + actf * cfg.w_sep * n_sep                               # separation cost
        dist_new = (self.pos - self.dest).to(F64).norm(dim=-1)
        w_fail = cfg.w_nav if cfg.w_fail_dist < 0 else cfg.w_fail_dist
        cost = cost + failing.to(F64) * (self.c_fail_i + w_fail * dist_new)     # Eq. (21), second branch
        ups_old = -cfg.w_nav * dist_old * (t_old[:, None] < self.Hdl)
        ups_new = -cfg.w_nav * dist_new * (self.t[:, None] < self.Hdl)
        reward = -cost + actf * (ups_new - ups_old)                              # Eqs. (25)-(26)

        fly = (act & self.present).to(F64)
        s_i = self.served_fraction()
        over = (self.D.gather(1, self.b) > self.K).to(F64)
        delivered = self.m_cur * s_i * self.rho_cur
        self.stat_cost += cost
        self.stat_viol += fly * over
        self.stat_out += fly * (delivered < cfg.R_min * (1 - 1e-9)).to(F64)
        self.stat_active_slots += fly
        self.stat_rate_ratio += fly * s_i
        self.stat_los += fly * (n_sep > 0).to(F64)
        if not self.mask_mode:
            d0 = (self.pos[:, :, None, :] == self.pos[:, None, :, :]).all(-1) & self.present[:, None, :] & ~self.eye
            self.stat_nmac += fly * d0.any(-1).to(F64)
        self.stat_nbr += torch.stack([(fly * self._nbr_cnt).sum(), (fly * (self._nbr_cnt > self.k)).sum()])
        u = self.D / self.K
        self.stat_load_jain += live * u.sum(1) ** 2 / (self.M * (u ** 2).sum(1) + 1e-12)
        self.stat_overload += live * (self.D > self.K).to(F64).mean(1)
        self.stat_potential += live * self.potential()
        self.stat_slots += live

        out = self._obs()
        out.update(reward=reward.to(F32), cost=cost.to(F32), active_before=act, penalty=g, n_sep=n_sep)
        return out

    # ----------------------------------------------------------- observation
    def _obs(self) -> Dict[str, torch.Tensor]:
        cfg, E, N, k = self.cfg, self.E, self.N, self.cap
        posf = self.pos.to(F32)
        gb = self.gbs_vox[self.b]                                               # (E,N,3)
        Db = self.D.gather(1, self.b)
        slack = (self.Hdl - self.t[:, None]).clamp(min=0) / max(1.0, float(self.Hdl.max()))
        parts = [posf / self.dims_norm, (gb - posf) / self.dims_norm,
                 (self.dest - self.pos).to(F32) / self.dims_norm,
                 (self.m_cur / self.m_max).to(F32)[..., None], (Db / self.K).to(F32)[..., None],
                 slack.to(F32)[..., None]]
        vel = ((self.pos - self.prev).to(F32) / cfg.v_max)
        if self.v2:
            parts.append(vel)
        own = torch.cat(parts, -1)
        if "own_pos" in cfg.ablate:
            own[..., 0:3] = 0.0
        if "vel" in cfg.ablate and self.v2:
            own[..., 12:15] = 0.0

        rel = posf[:, None, :, :] - posf[:, :, None, :]                          # [i, j] = q_j - q_i
        dist = rel.norm(dim=-1)
        nbr = (dist <= cfg.d_comm) & self.present[:, None, :] & ~self.eye
        same = (self.b[:, :, None] == self.b[:, None, :]).to(F32)
        mj = (self.m_cur / self.m_max).to(F32)[:, None, :].expand(E, N, N)
        fparts = [rel / cfg.d_comm, (dist / cfg.d_comm)[..., None], same[..., None], mj[..., None]]
        if self.v2:
            gbr = (gb[:, None, :, :] - posf[:, :, None, :]) / self.dims_norm   # g_{b_j} - q_i
            fparts += [vel[:, None].expand(E, N, N, 3), gbr]
        feat = torch.cat(fparts, -1)
        if "cocell" in cfg.ablate:
            feat[..., 4] = 0.0
        if "beta_nbr" in cfg.ablate:
            feat[..., 5] = 0.0
        if "vel" in cfg.ablate and self.v2:
            feat[..., 6:9] = 0.0
        nbrf = nbr.to(F32)
        cnt = nbrf.sum(-1, keepdim=True)
        self._nbr_cnt = cnt[..., 0].to(F64)
        mean = (feat * nbrf[..., None]).sum(2) / cnt.clamp_min(1.0)

        dkey = torch.where(nbr, dist, torch.full_like(dist, float("inf")))
        kk = min(k, N)
        order = torch.argsort(dkey, dim=-1, stable=True)[..., :kk]
        valid = nbr.gather(-1, order)
        knn = feat.gather(2, order[..., None].expand(E, N, kk, feat.shape[-1])) * valid[..., None]
        if kk < k:
            knn = torch.cat([knn, knn.new_zeros(E, N, k - kk, self.nbr_dim)], 2)
        knn = knn.reshape(E, N, k * self.nbr_dim)
        dem = (self.D / self.K).to(F32)[:, None, :].expand(E, N, self.M)
        obs = torch.cat([own, mean, cnt / 8.0, knn, dem], -1)
        return {"obs": obs, "mask": self.action_mask(), "active": self.active.clone(),
                "state": self.global_state()}

    def global_state(self) -> torch.Tensor:
        E, N = self.E, self.N
        slack = (self.Hdl - self.t[:, None]).clamp(min=0) / max(1.0, float(self.Hdl.max()))
        parts = [(self.pos.to(F32) / self.dims_norm).reshape(E, -1),
                 (self.dest.to(F32) / self.dims_norm).reshape(E, -1)]
        if self.v2:
            parts.append(((self.pos - self.prev).to(F32) / self.cfg.v_max).reshape(E, -1))
        parts += [slack.to(F32), self.active.to(F32)]
        pad = self.state_N - N
        if pad:
            parts = [torch.cat([x, x.new_zeros(E, pad * (x.shape[1] // N))], 1) for x in parts]
        return torch.cat(parts + [(self.D / self.K).to(F32)], 1)

    # ----------------------------------------------------------- statistics
    def episode_metrics(self) -> Dict[str, float]:
        succ = self.arrived
        T = torch.where(succ, self.T_arr, torch.zeros_like(self.T_arr))
        detour = (T / self.Tmin)[succ]
        util = torch.where(succ, self.Tmin / T.clamp_min(1), torch.zeros_like(T))
        jain_u = util.sum(1) ** 2 / (self.N * (util ** 2).sum(1) + 1e-12)
        act = self.stat_active_slots.clamp_min(1)
        slots = self.stat_slots.clamp_min(1)
        ratio = torch.where(succ, T / self.Tmin, torch.full_like(T, float("inf")))
        f = lambda x: float(x)
        out = {f"F@{x:.1f}": f((ratio <= x + 1e-9).to(F64).mean()) for x in CDF_GRID}
        A = act.sum()
        out.update({
            "success": f(succ.to(F64).mean()),
            "detour": f(detour.mean()) if detour.numel() else float("nan"),
            "p_out": f(self.stat_out.sum() / A),
            "jain_load": f((self.stat_load_jain / slots).mean()),
            "social_cost": f(self.stat_cost.sum(1).mean()),
            "J": f(self.stat_cost.mean()),
            "viol": f(self.stat_viol.sum() / A),
            "los": f(self.stat_los.sum() / A),
            "nmac": f(self.stat_nmac.sum() / A),
            "rate_ratio": f(self.stat_rate_ratio.sum() / A),
            "potential": f((self.stat_potential / slots).mean()),
            "overload": f((self.stat_overload / slots).mean()),
            "jain_util": f(jain_u.mean()),
            "ep_len": f(self.stat_slots.mean()),
            "route_ok": float(self.route_ok.mean()),
            "nbr_mean": f(self.stat_nbr[0] / A),
            "nbr_gt_k": f(self.stat_nbr[1] / A),
        })
        return out


def make_env(cfg: EnvConfig, n_envs: int, seed=None, backend: str = "numpy", device="cpu"):
    """LAWNEnv (NumPy, CPU) or TorchLAWNEnv (any torch device)."""
    if backend == "torch":
        return TorchLAWNEnv(cfg, n_envs=n_envs, seed=seed, device=device)
    return LAWNEnv(cfg, n_envs=n_envs, seed=seed)
