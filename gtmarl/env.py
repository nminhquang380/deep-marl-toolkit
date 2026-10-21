"""Vectorised multi-cell LAWN environment for the non-cooperative stochastic
congestion game G of the TWC manuscript (v22).

E independent episodes are simulated in lock-step. Association, average SNR,
SNR level, bearer size and ergodic per-RB rate are functions of the voxel
(Eqs. (1)-(6)) and are precomputed, so a slot costs a few array gathers.

Observation of UAV i (Eqs. (18), (27)-(29)), as one flat vector:
    [ nu_i (12) | xi_bar_i (6) | |M_i| (1) | nearest xi_ji (6 * nbr_cap) | {D_b / K_b} (B) ]
xi_bar_i is the mean aggregation of Eq. (29); the MLP f_omega of Eq. (30) lives
in the model. The nearest-neighbour block feeds the message MLP of gippo_msg and,
truncated to knn_k, the IPPO-MLP benchmark.
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
        raise ValueError("default GBS layout needs a square B; pass gbs_xy explicitly")
    xs = (np.arange(k) + 0.5) * cfg.W / k - 0.5
    ys = (np.arange(k) + 0.5) * cfg.R / k - 0.5
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


class ErgodicRate:
    """Spectral efficiency E[log2(1 + gamma |zeta|^2)] of Eq. (4), tabulated in gamma [dB]."""

    def __init__(self, fading: str, K_dB: float, n_samples: int = 20000, seed: int = 0):
        self.grid_dB = np.arange(-20.0, 70.0, 0.05)
        g = 10 ** (self.grid_dB / 10)
        if fading == "rician":
            K = 10 ** (K_dB / 10)
            rng = np.random.default_rng(seed)
            z = math.sqrt(K / (K + 1)) + math.sqrt(1 / (2 * (K + 1))) * (
                rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
            p = np.abs(z) ** 2
            self.table = np.array([np.log2(1 + gi * p).mean() for gi in g])
        elif fading == "none":
            self.table = np.log2(1 + g)
        else:
            raise ValueError(fading)

    def __call__(self, gamma_lin: np.ndarray) -> np.ndarray:
        return np.interp(10 * np.log10(gamma_lin), self.grid_dB, self.table)


class RadioMap:
    """Per-voxel association, average SNR, level, bearer and ergodic rate (Eqs. (1)-(6))."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        gxy = np.asarray(cfg.gbs_xy, dtype=np.float64) if cfg.gbs_xy is not None else default_gbs_xy(cfg)
        assert len(gxy) == cfg.M
        self.gbs_m = np.stack([(gxy[:, 0] + 0.5) * cfg.ds, (gxy[:, 1] + 0.5) * cfg.ds,
                               np.full(cfg.M, cfg.gbs_height)], axis=1)
        self.gbs_vox = np.stack([gxy[:, 0], gxy[:, 1],
                                 np.full(cfg.M, (cfg.gbs_height - cfg.h_min) / cfg.ds)], axis=1)

        ix, iy, iz = np.meshgrid(np.arange(cfg.W), np.arange(cfg.R), np.arange(cfg.H), indexing="ij")
        vox_m = np.stack([(ix + 0.5) * cfg.ds, (iy + 0.5) * cfg.ds, cfg.h_min + iz * cfg.ds], axis=-1)
        dist = np.linalg.norm(vox_m[..., None, :] - self.gbs_m, axis=-1)       # (W,R,H,B)
        self.serve = dist.argmin(-1)                                            # Eq. (1), ties -> smallest index
        d = np.take_along_axis(dist, self.serve[..., None], -1)[..., 0]
        p_tx = 10 ** ((cfg.P_tx_dBm - 30) / 10)
        noise = 10 ** ((cfg.noise_dBm - 30) / 10)
        g0 = 10 ** (cfg.G0_dB / 10)
        self.snr_lin = p_tx * g0 / (noise * d ** cfg.alpha)                     # Eq. (3), average SNR
        self.snr_dB = 10 * np.log10(self.snr_lin)

        self.se = ErgodicRate(cfg.fading, cfg.rician_K_dB)
        th = np.asarray(cfg.thresholds(), dtype=np.float64)
        assert np.all(np.diff(th) < 0), "thresholds must be strictly decreasing"
        self.th_dB = th
        self.th_lin = 10 ** (th / 10)
        self.L = len(th)
        self.rho_bar = cfg.delta_f * self.se(self.th_lin)                       # guaranteed per-RB rate
        self.m_of_level = np.ceil(cfg.R_min / self.rho_bar).astype(np.int64)    # Eq. (6)
        self.level = self.level_of(self.snr_lin)                                # Eq. (5), 0-based
        self.m = self.m_of_level[self.level]
        self.m_max = int(self.m_of_level.max())
        self.rho_vox = cfg.delta_f * self.se(self.snr_lin)                      # ergodic per-RB rate rho(gamma_i)
        self.min_snr_dB = float(self.snr_dB.min())
        self.covered = bool(self.min_snr_dB >= th[-1])

    def level_of(self, snr_lin: np.ndarray) -> np.ndarray:
        lvl = (snr_lin[..., None] < self.th_lin).sum(-1)
        return np.minimum(lvl, self.L - 1)

    def summary(self) -> Dict:
        lv, cnt = np.unique(self.level, return_counts=True)
        return {
            "snr_dB_range": (round(self.min_snr_dB, 2), round(float(self.snr_dB.max()), 2)),
            "covered": self.covered,
            "thresholds_dB": self.th_dB.tolist(),
            "rho_bar_Mbps": [round(x / 1e6, 3) for x in self.rho_bar],
            "bearer_RBs_per_level": self.m_of_level.tolist(),
            "voxel_share_per_level": {int(l) + 1: round(float(c) / self.level.size, 3) for l, c in zip(lv, cnt)},
        }


class LAWNEnv:
    """E parallel episodes of the Markov game G (Eq. (17))."""

    def __init__(self, cfg: EnvConfig, n_envs: int = 1, seed: Optional[int] = None):
        if cfg.d_comm < cfg.d_safe + 2 * cfg.v_max:
            raise ValueError("Eq. (19) needs d_comm >= d_safe + 2 v_max")
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
        self.k = cfg.knn_k
        self.cap = max(cfg.nbr_cap, cfg.knn_k)
        self.obs_dim = OWN_DIM + GRAPH_DIM + NBR_DIM * self.cap + self.M
        self.state_dim = 3 * self.N + 3 * self.N + self.N + self.N + self.M
        self._all_vox = np.stack(np.meshgrid(np.arange(cfg.W), np.arange(cfg.R), np.arange(cfg.H),
                                             indexing="ij"), -1).reshape(-1, 3)
        self._origin_pool = self._make_origin_pool()
        self.tries = 40

    # ------------------------------------------------------------------ reset
    def _make_origin_pool(self):
        cfg, V = self.cfg, self._all_vox
        if cfg.origin_mode == "uniform":
            return [np.arange(len(V))]
        pools = []
        for fx, fy in cfg.depots:
            c = np.array([fx * cfg.W - 0.5, fy * cfg.R - 0.5])
            ok = (np.linalg.norm(V[:, :2] - c, axis=1) <= cfg.depot_radius) & (V[:, 2] < cfg.depot_layers)
            pools.append(np.flatnonzero(ok))
        return pools

    def _cell_ok(self, vox: np.ndarray) -> bool:
        b = self.radio.serve[tuple(vox.T)]
        m = self.radio.m[tuple(vox.T)]
        return not np.any(np.bincount(b, weights=m, minlength=self.M) > self.K)

    def _sample_origins(self):
        N, V, sep = self.N, self._all_vox, self.sep_move
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
                raise RuntimeError("depot too small for N; increase depot_radius / depot_layers")
        return np.asarray(orig)[self.rng.permutation(N)]

    def _sample_dests(self, orig):
        cfg, N, V, sep = self.cfg, self.N, self._all_vox, self.sep_move
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

    def _sample_episode(self):
        """Returns (origins, destinations, route_ok)."""
        mode = self.cfg.init_check
        if mode == "off":
            while True:
                o = self._sample_origins(); d = self._sample_dests(o)
                if d is not None:
                    return o, d, self._cell_ok(o) and self._cell_ok(d)
        for _ in range(self.tries):                            # strict: Sec. II-A
            o = self._sample_origins()
            if not self._cell_ok(o):
                continue
            d = self._sample_dests(o)
            if d is not None and self._cell_ok(d):
                return o, d, True
        if mode == "strict":
            raise RuntimeError("no episode with non-overloaded initial and final configurations; "
                               "reduce N or use init_check=soft")
        for _ in range(self.tries):                            # soft: origins only
            o = self._sample_origins()
            if self._cell_ok(o):
                d = self._sample_dests(o)
                if d is not None:
                    return o, d, False
        while True:
            o = self._sample_origins(); d = self._sample_dests(o)
            if d is not None:
                return o, d, False

    def reset(self) -> Dict[str, np.ndarray]:
        E, N = self.E, self.N
        self.pos = np.zeros((E, N, 3), np.int64)
        self.dest = np.zeros((E, N, 3), np.int64)
        self.route_ok = np.zeros(E, bool)
        for e in range(E):
            self.pos[e], self.dest[e], self.route_ok[e] = self._sample_episode()
        self.orig = self.pos.copy()
        self.Tmin = np.ceil(np.abs(self.pos - self.dest).sum(-1) / self.cfg.v_max).astype(np.int64)
        self.Hdl = np.floor(self.cfg.eta * self.Tmin + 1e-9).astype(np.int64)    # T_i^max
        self.t = np.zeros(E, np.int64)
        self.arrived = np.zeros((E, N), bool)
        self.failed = np.zeros((E, N), bool)
        self.T_arr = np.full((E, N), -1, np.int64)
        self._radio_update()
        self.stat_cost = np.zeros((E, N))
        self.stat_viol = np.zeros((E, N))
        self.stat_out = np.zeros((E, N))
        self.stat_active_slots = np.zeros((E, N))
        self.stat_rate_ratio = np.zeros((E, N))
        self.stat_load_jain = np.zeros(E)
        self.stat_overload = np.zeros(E)
        self.stat_potential = np.zeros(E)
        self.stat_slots = np.zeros(E)
        self.stat_nbr = np.zeros(2)                       # sum |M_i|, #(|M_i| > k) over in-flight slots
        self.traj = [self.pos.copy()]
        self.D_hist = [self.D.copy()]
        return self._obs()

    # -------------------------------------------------------------- dynamics
    @property
    def active(self) -> np.ndarray:
        return ~(self.arrived | self.failed)

    @property
    def done(self) -> np.ndarray:
        return ~self.active.any(axis=1)

    def _radio_update(self):
        """Association, bearer and cell demand at the current positions (Eqs. (1), (5)-(7))."""
        idx = tuple(np.moveaxis(self.pos, -1, 0))
        self.b = self.radio.serve[idx]
        self.m_cur = self.radio.m[idx]
        self.rho_cur = self.radio.rho_vox[idx]
        self.D = np.zeros((self.E, self.M))
        np.add.at(self.D, (np.repeat(np.arange(self.E), self.N), self.b.ravel()), self.m_cur.ravel())

    def served_fraction(self) -> np.ndarray:
        """s_b of Eq. (8) for every UAV's cell, shape (E, N)."""
        s = np.minimum(1.0, self.K / np.maximum(self.D, 1e-9))
        return np.take_along_axis(s, self.b, axis=1)

    def penalty(self) -> np.ndarray:
        """Congestion penalty at the current state."""
        pen = self.cfg.penalty
        if pen == "lost":                                                        # beta_i - kappa_i
            return self.m_cur * (1.0 - self.served_fraction())
        if pen == "shortfall":                                                   # [R_min - R_i]^+ / R_min = 1 - s_b
            return 1.0 - self.served_fraction()
        if pen == "indicator":                                                   # 1[R_i < R_min] = 1[D_b > K_b]
            return (np.take_along_axis(self.D, self.b, axis=1) > self.K).astype(float)
        over = np.maximum(self.D - self.K, 0.0)                                  # Eq. (14)
        return np.minimum(self.m_cur, np.take_along_axis(over, self.b, axis=1))

    def potential(self) -> np.ndarray:
        """phi = mu * sum_b [D_b - K_b]^+, Eq. (24) (per env)."""
        return self.cfg.mu * np.maximum(self.D - self.K, 0.0).sum(axis=1)

    def action_mask(self) -> np.ndarray:
        """Admissible set A_i(x), Eq. (19). Shape (E, N, A), bool."""
        land = self.pos[:, :, None, :] + self.acts[None, None]                   # (E,N,A,3)
        inb = np.all((land >= 0) & (land < self.dims), axis=-1)
        cur = np.linalg.norm(self.pos[:, :, None, :] - self.pos[:, None, :, :], axis=-1)  # (E,N,N)
        nbr = cur <= self.cfg.d_comm
        nbr[:, np.arange(self.N), np.arange(self.N)] = False
        dist = np.linalg.norm(land[:, :, :, None, :] - self.pos[:, None, None, :, :], axis=-1)
        dist = np.where(nbr[:, :, None, :], dist, np.inf)
        mask = inb & (dist.min(axis=-1) >= self.sep_move)
        mask[..., 0] = True
        act = self.active
        mask[~act] = False
        mask[~act, 0] = True
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

        dist_old = np.linalg.norm(self.pos - self.dest, axis=-1)
        self.pos = self.pos + self.acts[actions] * act[..., None]
        self.t = self.t + live_env.astype(np.int64)
        self._radio_update()                                                     # landing state x'

        g = self.penalty()
        arrived_now = act & np.all(self.pos == self.dest, axis=-1)
        cost = act * (1.0 / self.Tmin + cfg.mu * g)                              # Eq. (21), third branch
        failing = act & ~arrived_now & (self.t[:, None] >= self.Hdl)
        dist_new = np.linalg.norm(self.pos - self.dest, axis=-1)
        cost = cost + failing * (cfg.c_fail + cfg.w_fail_dist * dist_new)        # Eq. (21), second branch

        ups_old = -cfg.w_nav * dist_old * (t_old[:, None] < self.Hdl)
        ups_new = -cfg.w_nav * dist_new * (self.t[:, None] < self.Hdl)
        reward = -cost + act * (ups_new - ups_old)                               # Eqs. (25)-(26)

        self.T_arr[arrived_now] = self.t[np.nonzero(arrived_now)[0]]
        self.arrived |= arrived_now
        self.failed |= failing

        # statistics at the landing state (in-flight UAV-slots = act)
        s_i = self.served_fraction()
        over = np.take_along_axis(self.D, self.b, axis=1) > self.K
        delivered = self.m_cur * s_i * self.rho_cur                              # kappa_i * rho(gamma_i)
        self.stat_cost += cost
        self.stat_viol += act * over                                             # R_i < R_min, Eq. (9)
        self.stat_out += act * (delivered < cfg.R_min * (1 - 1e-9))              # outage, Sec. V-A
        self.stat_active_slots += act
        self.stat_rate_ratio += act * s_i
        if hasattr(self, "_nbr_cnt"):
            self.stat_nbr += [(act * self._nbr_cnt).sum(), (act * (self._nbr_cnt > self.k)).sum()]
        u = self.D / self.K
        self.stat_load_jain += live_env * u.sum(1) ** 2 / (self.M * (u ** 2).sum(1) + 1e-12)
        self.stat_overload += live_env * (self.D > self.K).mean(1)
        self.stat_potential += live_env * self.potential()
        self.stat_slots += live_env
        self.traj.append(self.pos.copy())
        self.D_hist.append(self.D.copy())

        out = self._obs()
        out.update(reward=reward.astype(np.float32), cost=cost.astype(np.float32),
                   active_before=act, penalty=g)
        return out

    # ----------------------------------------------------------- observation
    def _obs(self) -> Dict[str, np.ndarray]:
        cfg, N, k = self.cfg, self.N, self.cap
        posf = self.pos.astype(np.float32)
        gb = self.radio.gbs_vox[self.b].astype(np.float32)
        Db = np.take_along_axis(self.D, self.b, axis=1)
        slack = (self.Hdl - self.t[:, None]).clip(min=0) / max(1, int(self.Hdl.max()))
        own = np.concatenate([
            posf / self.dims_norm,
            (gb - posf) / self.dims_norm,
            (self.dest - self.pos) / self.dims_norm,
            (self.m_cur / self.radio.m_max)[..., None],
            (Db / self.K)[..., None],
            slack[..., None],
        ], axis=-1)                                                              # Eq. (27)
        if "own_pos" in cfg.ablate:
            own[..., 0:3] = 0.0

        rel = posf[:, None, :, :] - posf[:, :, None, :]                          # q_j - q_i
        dist = np.linalg.norm(rel, axis=-1)
        nbr = dist <= cfg.d_comm
        nbr[:, np.arange(N), np.arange(N)] = False
        same = (self.b[:, :, None] == self.b[:, None, :]).astype(np.float32)
        mj = np.broadcast_to((self.m_cur / self.radio.m_max)[:, None, :], same.shape)
        feat = np.concatenate([rel / cfg.d_comm, (dist / cfg.d_comm)[..., None],
                               same[..., None], mj[..., None]], axis=-1)        # Eq. (28)
        if "cocell" in cfg.ablate:
            feat[..., 4] = 0.0
        if "beta_nbr" in cfg.ablate:
            feat[..., 5] = 0.0
        cnt = nbr.sum(-1, keepdims=True).astype(np.float32)
        self._nbr_cnt = cnt[..., 0]
        mean = (feat * nbr[..., None]).sum(2) / np.maximum(cnt, 1.0)             # Eq. (29)

        # nearest neighbours within d_comm, sorted by distance and zero-padded to nbr_cap
        # (message passing of gippo_msg; the first knn_k entries feed IPPO-MLP)
        dkey = np.where(nbr, dist, np.inf)
        order = np.argsort(dkey, axis=-1)[..., :k]                               # (E,N,k)
        valid = np.take_along_axis(nbr, order, axis=-1)
        knn = np.take_along_axis(feat, order[..., None], axis=2) * valid[..., None]
        if knn.shape[2] < k:
            knn = np.concatenate([knn, np.zeros(knn.shape[:2] + (k - knn.shape[2], NBR_DIM), knn.dtype)], 2)
        knn = knn.reshape(self.E, N, k * NBR_DIM)

        dem = np.broadcast_to((self.D / self.K)[:, None, :], (self.E, N, self.M))
        obs = np.concatenate([own, mean, cnt / 8.0, knn, dem], axis=-1).astype(np.float32)
        return {"obs": obs, "mask": self.action_mask(), "active": self.active.copy(),
                "state": self.global_state()}

    def global_state(self) -> np.ndarray:
        """Joint state for the CTDE-MAPPO benchmark."""
        posf = (self.pos / self.dims_norm).reshape(self.E, -1)
        dstf = (self.dest / self.dims_norm).reshape(self.E, -1)
        slack = (self.Hdl - self.t[:, None]).clip(min=0) / max(1, int(self.Hdl.max()))
        return np.concatenate([posf, dstf, slack, self.active.astype(np.float64), self.D / self.K],
                              axis=1).astype(np.float32)

    # ----------------------------------------------------------- statistics
    def episode_metrics(self) -> Dict[str, float]:
        """Metrics of Sec. V-A over the E finished episodes."""
        succ = self.arrived
        T = np.where(succ, self.T_arr, 0)
        detour = (T / self.Tmin)[succ]
        util = np.where(succ, self.Tmin / np.maximum(T, 1), 0.0)
        jain_u = util.sum(1) ** 2 / (self.N * (util ** 2).sum(1) + 1e-12)
        act = np.maximum(self.stat_active_slots, 1)
        slots = np.maximum(self.stat_slots, 1)
        return {
            "success": float(succ.mean()),                       # MSR
            "detour": float(detour.mean()) if detour.size else float("nan"),
            "p_out": float(self.stat_out.sum() / act.sum()),     # outage probability
            "jain_load": float((self.stat_load_jain / slots).mean()),   # Jain of cell utilisations
            "social_cost": float(self.stat_cost.sum(1).mean()),  # sum_i J_i
            "J": float(self.stat_cost.mean()),
            "viol": float(self.stat_viol.sum() / act.sum()),     # share of in-flight slots with R_i < R_min
            "rate_ratio": float(self.stat_rate_ratio.sum() / act.sum()),
            "potential": float((self.stat_potential / slots).mean()),
            "overload": float((self.stat_overload / slots).mean()),
            "jain_util": float(jain_u.mean()),
            "ep_len": float(self.stat_slots.mean()),
            "route_ok": float(self.route_ok.mean()),            # share of episodes meeting Sec. II-A
            "nbr_mean": float(self.stat_nbr[0] / act.sum()),   # mean |M_i| of in-flight UAVs
            "nbr_gt_k": float(self.stat_nbr[1] / act.sum()),   # share with more than k neighbours
        }
