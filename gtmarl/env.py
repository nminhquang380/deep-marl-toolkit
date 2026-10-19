"""Vectorised multi-cell LAWN environment for the non-cooperative stochastic
congestion game G of the TWC manuscript (v18).

E independent episodes are simulated in lock-step. Per-voxel association,
SNR level and bearer size are precomputed (Eqs. (1)-(4)), so a slot costs a
few array gathers regardless of N.

Observation of UAV i (Eqs. (16), (26)-(28)), as one flat vector:
    [ nu_i (12) | xi_bar_i (6) | |M_i| (1) | {D_b / K_b}_{b in B} (M) ]
The neighbour summary xi_bar_i is the mean of the raw neighbour features of
Eq. (27), i.e. the aggregation step of Eq. (28); the MLP f_omega of Eq. (29)
lives in the model.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .config import EnvConfig

OWN_DIM = 12
NBR_DIM = 6
GRAPH_DIM = NBR_DIM + 1


def action_set(v_max: int) -> np.ndarray:
    """All displacements with ||a||_1 <= v_max; index 0 is hover."""
    acts = [(0, 0, 0)]
    r = range(-v_max, v_max + 1)
    for dx in r:
        for dy in r:
            for dz in r:
                if (dx, dy, dz) != (0, 0, 0) and abs(dx) + abs(dy) + abs(dz) <= v_max:
                    acts.append((dx, dy, dz))
    return np.asarray(acts, dtype=np.int64)


def default_gbs_xy(cfg: EnvConfig) -> np.ndarray:
    k = int(round(math.sqrt(cfg.M)))
    if k * k != cfg.M:
        raise ValueError("default GBS layout needs a square M; pass gbs_xy explicitly")
    xs = (np.arange(k) + 0.5) * cfg.W / k - 0.5
    ys = (np.arange(k) + 0.5) * cfg.R / k - 0.5
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


class RadioMap:
    """Per-voxel association, SNR, level and bearer tables (Eqs. (1)-(4))."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        gxy = np.asarray(cfg.gbs_xy, dtype=np.float64) if cfg.gbs_xy is not None else default_gbs_xy(cfg)
        assert len(gxy) == cfg.M
        # GBS positions in metres and in continuous voxel coordinates
        self.gbs_m = np.stack([(gxy[:, 0] + 0.5) * cfg.ds, (gxy[:, 1] + 0.5) * cfg.ds,
                               np.full(cfg.M, cfg.gbs_height)], axis=1)
        self.gbs_vox = np.stack([gxy[:, 0], gxy[:, 1],
                                 np.full(cfg.M, (cfg.gbs_height - cfg.h_min) / cfg.ds)], axis=1)

        ix, iy, iz = np.meshgrid(np.arange(cfg.W), np.arange(cfg.R), np.arange(cfg.H), indexing="ij")
        vox_m = np.stack([(ix + 0.5) * cfg.ds, (iy + 0.5) * cfg.ds, cfg.h_min + iz * cfg.ds], axis=-1)
        dist = np.linalg.norm(vox_m[..., None, :] - self.gbs_m, axis=-1)       # (W,R,H,M)
        self.serve = dist.argmin(-1)                                            # Eq. (1)
        d = np.take_along_axis(dist, self.serve[..., None], -1)[..., 0]
        p_tx = 10 ** ((cfg.P_tx_dBm - 30) / 10)
        noise = 10 ** ((cfg.noise_dBm - 30) / 10)
        beta0 = 10 ** (cfg.beta0_dB / 10)
        self.snr_lin = p_tx * beta0 / (noise * d ** cfg.alpha)                  # Eq. (2)
        self.snr_dB = 10 * np.log10(self.snr_lin)

        th = np.asarray(cfg.snr_th_dB, dtype=np.float64)
        assert np.all(np.diff(th) < 0), "thresholds must be strictly decreasing"
        self.th_lin = 10 ** (th / 10)
        self.L = len(th)
        self.rho_bar = cfg.delta_f * np.log2(1 + self.th_lin)                   # rate per RB at level l
        self.m_of_level = np.ceil(cfg.R_min / self.rho_bar).astype(np.int64)    # Eq. (4)
        self.level = self.level_of(self.snr_lin)                                # Eq. (3), 0-based
        self.m = self.m_of_level[self.level]
        self.m_max = int(self.m_of_level.max())
        self.min_snr_dB = float(self.snr_dB.min())
        self.covered = bool(self.min_snr_dB >= th[-1])

    def level_of(self, snr_lin: np.ndarray) -> np.ndarray:
        # number of thresholds strictly above snr -> 0-based level; clamp to L-1 (coverage)
        lvl = (snr_lin[..., None] < self.th_lin).sum(-1)
        return np.minimum(lvl, self.L - 1)

    def summary(self) -> Dict:
        lv, cnt = np.unique(self.level, return_counts=True)
        return {
            "snr_dB_range": (round(self.min_snr_dB, 2), round(float(self.snr_dB.max()), 2)),
            "covered": self.covered,
            "bearer_RBs_per_level": self.m_of_level.tolist(),
            "voxel_share_per_level": {int(l) + 1: round(c / self.level.size, 3) for l, c in zip(lv, cnt)},
        }


class LAWNEnv:
    """E parallel episodes of the Markov game G (Eq. (15))."""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: Optional[int] = None):
        if cfg.d_comm < cfg.d_safe + 2 * cfg.v_max:
            raise ValueError("Eq. (17) needs d_comm >= d_safe + 2 v_max")
        self.cfg = cfg
        self.E = n_envs
        self.N = cfg.N
        self.M = cfg.M
        self.K = cfg.effective_K()
        self.radio = RadioMap(cfg)
        self.acts = action_set(cfg.v_max)
        self.A = len(self.acts)
        self.dims = np.array([cfg.W, cfg.R, cfg.H])
        self.dims_norm = np.maximum(self.dims - 1, 1).astype(np.float32)
        self.sep_move = cfg.d_safe + cfg.v_max
        self.rng = np.random.default_rng(seed)
        self.rician_K = 10 ** (cfg.rician_K_dB / 10)
        self.obs_dim = OWN_DIM + GRAPH_DIM + self.M
        self.state_dim = 3 * self.N + 3 * self.N + self.N + self.N + self.M

    # ------------------------------------------------------------------ reset
    def _sample_episode(self):
        cfg, N = self.cfg, self.N
        sep = self.sep_move
        all_vox = np.stack(np.meshgrid(np.arange(cfg.W), np.arange(cfg.R), np.arange(cfg.H),
                                       indexing="ij"), -1).reshape(-1, 3)
        for _ in range(1000):
            # origins: pairwise >= d_safe + v_max (separated, Eq. (9)), no overloaded cell
            perm = self.rng.permutation(len(all_vox))
            orig = []
            for idx in perm:
                v = all_vox[idx]
                if not orig or np.min(np.linalg.norm(np.asarray(orig) - v, axis=1)) >= sep:
                    orig.append(v)
                    if len(orig) == N:
                        break
            if len(orig) < N:
                continue
            orig = np.asarray(orig)
            if cfg.init_no_overload:
                b = self.radio.serve[tuple(orig.T)]
                m = self.radio.m[tuple(orig.T)]
                if np.any(np.bincount(b, weights=m, minlength=self.M) > self.K):
                    continue
            # destinations: L1 >= min_od from own origin, >= sep from all origins and
            # other destinations (an arrived UAV hovers at its destination)
            dmin_orig = np.min(np.linalg.norm(all_vox[:, None, :] - orig[None], axis=-1), axis=1)
            dest, ok = [], True
            free = dmin_orig >= sep                     # >= sep from all origins and chosen dests
            for i in range(N):
                cand = free & (np.abs(all_vox - orig[i]).sum(1) >= cfg.min_od_l1)
                idx = np.flatnonzero(cand)
                if len(idx) == 0:
                    ok = False
                    break
                v = all_vox[self.rng.choice(idx)]
                dest.append(v)
                free &= np.linalg.norm(all_vox - v, axis=1) >= sep
            if ok and cfg.init_no_overload:
                # route condition (Sec. II-A): the final configuration overloads no cell either
                dest_a = np.asarray(dest)
                b = self.radio.serve[tuple(dest_a.T)]
                m = self.radio.m[tuple(dest_a.T)]
                ok = not np.any(np.bincount(b, weights=m, minlength=self.M) > self.K)
            if ok:
                return orig, np.asarray(dest)
        raise RuntimeError("could not sample a valid episode; enlarge the grid or reduce N")

    def reset(self) -> Dict[str, np.ndarray]:
        E, N = self.E, self.N
        self.pos = np.zeros((E, N, 3), np.int64)
        self.dest = np.zeros((E, N, 3), np.int64)
        for e in range(E):
            self.pos[e], self.dest[e] = self._sample_episode()
        self.orig = self.pos.copy()
        self.Tmin = np.ceil(np.abs(self.pos - self.dest).sum(-1) / self.cfg.v_max).astype(np.int64)
        self.Hdl = np.floor(self.cfg.eta * self.Tmin + 1e-9).astype(np.int64)    # H_i
        self.Hmax = self.Hdl.max(axis=1)                                           # per env
        self.t = np.zeros(E, np.int64)
        self.arrived = np.zeros((E, N), bool)
        self.failed = np.zeros((E, N), bool)
        self.T_arr = np.full((E, N), -1, np.int64)
        self._radio_update()
        # episode statistics
        self.stat_cost = np.zeros((E, N))
        self.stat_viol = np.zeros((E, N))
        self.stat_active_slots = np.zeros((E, N))
        self.stat_rate_ratio = np.zeros((E, N))
        self.stat_load_jain = np.zeros(E)
        self.stat_overload = np.zeros(E)
        self.stat_slots = np.zeros(E)
        self.traj = [self.pos.copy()]
        return self._obs()

    # -------------------------------------------------------------- dynamics
    @property
    def active(self) -> np.ndarray:
        return ~(self.arrived | self.failed)

    @property
    def done(self) -> np.ndarray:
        return ~self.active.any(axis=1)

    def _radio_update(self):
        """Association, bearer and cell demand at the current positions."""
        idx = tuple(np.moveaxis(self.pos, -1, 0))
        self.b = self.radio.serve[idx]
        if self.cfg.fading == "rician":
            Kr = self.rician_K
            h = math.sqrt(Kr / (Kr + 1)) + math.sqrt(1 / (2 * (Kr + 1))) * (
                self.rng.standard_normal(self.b.shape) + 1j * self.rng.standard_normal(self.b.shape))
            snr = self.radio.snr_lin[idx] * np.abs(h) ** 2
            self.m_cur = self.radio.m_of_level[self.radio.level_of(snr)]
        else:
            self.m_cur = self.radio.m[idx]
        self.D = np.zeros((self.E, self.M))
        np.add.at(self.D, (np.repeat(np.arange(self.E), self.N), self.b.ravel()), self.m_cur.ravel())

    def penalty(self) -> np.ndarray:
        """g_i at the current state, Eq. (12): min{m_i, [D_b - K_b]^+}."""
        over = np.maximum(self.D - self.K, 0.0)
        return np.minimum(self.m_cur, np.take_along_axis(over, self.b, axis=1))

    def potential(self) -> np.ndarray:
        """phi = mu * sum_b [D_b - K_b]^+, Eq. (22) (per env)."""
        return self.cfg.mu * np.maximum(self.D - self.K, 0.0).sum(axis=1)

    def action_mask(self) -> np.ndarray:
        """Admissible set A_i(x), Eq. (17). Shape (E, N, A), bool."""
        land = self.pos[:, :, None, :] + self.acts[None, None]                   # (E,N,A,3)
        inb = np.all((land >= 0) & (land < self.dims), axis=-1)
        diff = land[:, :, :, None, :] - self.pos[:, None, None, :, :]            # (E,N,A,N,3)
        dist = np.linalg.norm(diff, axis=-1)
        cur = np.linalg.norm(self.pos[:, :, None, :] - self.pos[:, None, :, :], axis=-1)  # (E,N,N)
        nbr = (cur <= self.cfg.d_comm)
        nbr[:, np.arange(self.N), np.arange(self.N)] = False
        dist = np.where(nbr[:, :, None, :], dist, np.inf)
        clear = dist.min(axis=-1) >= self.sep_move
        mask = inb & clear
        mask[..., 0] = True                                                      # hover always admissible
        act = self.active
        mask[~act] = False
        mask[~act, 0] = True                                                     # inactive UAVs hover
        return mask

    def step(self, actions: np.ndarray, check_mask: bool = False) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        actions = np.asarray(actions, np.int64)
        if check_mask:
            m = self.action_mask()
            assert np.take_along_axis(m, actions[..., None], -1).all(), "inadmissible action"
        act = self.active.copy()
        live_env = ~self.done
        t_old = self.t.copy()

        disp = self.acts[actions] * act[..., None]
        dist_old = np.linalg.norm(self.pos - self.dest, axis=-1)
        self.pos = self.pos + disp
        self.t = self.t + live_env.astype(np.int64)
        self._radio_update()                                                     # landing state x'

        g = self.penalty()
        arrived_now = act & np.all(self.pos == self.dest, axis=-1)
        cost = act * (1.0 / self.Tmin + cfg.mu * g)                              # Eq. (19), third branch
        failing = act & ~arrived_now & (self.t[:, None] >= self.Hdl)
        cost = cost + failing * cfg.rho_fail                                     # Eq. (19), second branch

        dist_new = np.linalg.norm(self.pos - self.dest, axis=-1)
        ups_old = -cfg.rho_nav * dist_old * (t_old[:, None] < self.Hdl)
        ups_new = -cfg.rho_nav * dist_new * (self.t[:, None] < self.Hdl)
        r_nav = act * (ups_new - ups_old)                                        # Eq. (24)
        reward = -cost + r_nav                                                   # Eq. (25)

        self.T_arr[arrived_now] = self.t[np.nonzero(arrived_now)[0]]
        self.arrived |= arrived_now
        self.failed |= failing

        # statistics (rate of Eq. (7): R_i / R_min = s_b)
        s = np.minimum(1.0, self.K / np.maximum(self.D, 1e-9))
        s_i = np.take_along_axis(s, self.b, axis=1)
        self.stat_cost += cost
        self.stat_viol += act * (g > 0)
        self.stat_active_slots += act
        self.stat_rate_ratio += act * s_i
        u = self.D / self.K
        jain = u.sum(1) ** 2 / (self.M * (u ** 2).sum(1) + 1e-12)
        self.stat_load_jain += live_env * jain
        self.stat_overload += live_env * (self.D > self.K).mean(1)
        self.stat_slots += live_env
        self.traj.append(self.pos.copy())

        out = self._obs()
        out.update(reward=reward.astype(np.float32), cost=cost.astype(np.float32),
                   active_before=act, penalty=g)
        return out

    # ----------------------------------------------------------- observation
    def _obs(self) -> Dict[str, np.ndarray]:
        cfg, N = self.cfg, self.N
        posf = self.pos.astype(np.float32)
        gb = self.radio.gbs_vox[self.b].astype(np.float32)                       # serving GBS (E,N,3)
        Db = np.take_along_axis(self.D, self.b, axis=1)
        slack = (self.Hdl - self.t[:, None]).clip(min=0) / max(1, int(self.Hdl.max()))
        own = np.concatenate([
            posf / self.dims_norm,
            (gb - posf) / self.dims_norm,
            (self.dest - self.pos) / self.dims_norm,
            (self.m_cur / self.radio.m_max)[..., None],
            (Db / self.K)[..., None],
            slack[..., None],
        ], axis=-1)                                                              # Eq. (26)

        rel = posf[:, None, :, :] - posf[:, :, None, :]                          # q_j - q_i, (E,N,N,3)
        dist = np.linalg.norm(rel, axis=-1)
        nbr = dist <= cfg.d_comm
        nbr[:, np.arange(N), np.arange(N)] = False
        same = (self.b[:, :, None] == self.b[:, None, :]).astype(np.float32)
        mj = np.broadcast_to((self.m_cur / self.radio.m_max)[:, None, :], same.shape)
        feat = np.concatenate([rel / cfg.d_comm, (dist / cfg.d_comm)[..., None],
                               same[..., None], mj[..., None]], axis=-1)        # Eq. (27)
        cnt = nbr.sum(-1, keepdims=True).astype(np.float32)
        mean = (feat * nbr[..., None]).sum(2) / np.maximum(cnt, 1.0)             # Eq. (28)
        cnt_norm = cnt / 8.0
        dem = np.broadcast_to((self.D / self.K)[:, None, :], (self.E, N, self.M))
        obs = np.concatenate([own, mean, cnt_norm, dem], axis=-1).astype(np.float32)
        return {"obs": obs, "mask": self.action_mask(), "active": self.active.copy(),
                "state": self.global_state()}

    def global_state(self) -> np.ndarray:
        """Joint state for the MAPPO (centralised critic) baseline."""
        posf = (self.pos / self.dims_norm).reshape(self.E, -1)
        dstf = (self.dest / self.dims_norm).reshape(self.E, -1)
        slack = (self.Hdl - self.t[:, None]).clip(min=0) / max(1, int(self.Hdl.max()))
        return np.concatenate([posf, dstf, slack, self.active.astype(np.float64), self.D / self.K],
                              axis=1).astype(np.float32)

    # ----------------------------------------------------------- statistics
    def episode_metrics(self) -> Dict[str, float]:
        """Metrics over the E finished episodes."""
        succ = self.arrived
        T = np.where(succ, self.T_arr, 0)
        detour = (T / self.Tmin)[succ]
        util = np.where(succ, self.Tmin / np.maximum(T, 1), 0.0)
        jain_u = util.sum(1) ** 2 / (self.N * (util ** 2).sum(1) + 1e-12)
        act = np.maximum(self.stat_active_slots, 1)
        return {
            "success": float(succ.mean()),
            "detour": float(detour.mean()) if detour.size else float("nan"),
            "J": float(self.stat_cost.mean()),                 # mean individual game cost J_i
            "welfare": float(self.stat_cost.sum(1).mean()),   # sum_i J_i (lower is better)
            "viol": float(self.stat_viol.sum() / act.sum()),  # share of active slots with R_i < R_min
            "rate_ratio": float(self.stat_rate_ratio.sum() / act.sum()),
            "jain_util": float(jain_u.mean()),
            "jain_load": float((self.stat_load_jain / np.maximum(self.stat_slots, 1)).mean()),
            "overload": float((self.stat_overload / np.maximum(self.stat_slots, 1)).mean()),
            "ep_len": float(self.stat_slots.mean()),
        }
