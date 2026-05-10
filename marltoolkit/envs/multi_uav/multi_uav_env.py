import math
import argparse
from typing import List, Optional, Tuple, Union

import numpy as np


class MultiUAVEnv:
    """Multi-UAV trajectory optimization environment for cooperative MARL.

    N UAVs navigate from randomized origins to randomized destinations in a 3D
    voxel grid while maximizing uplink communication rate to M ground base
    stations (GBSs).  GBSs allocate radio blocks equally among all UAVs they
    serve, creating coupling: joining a congested cell hurts co-served UAVs.

    The environment implements the same duck-typed interface as SMACWrapperEnv
    so it works directly with the existing episode_runner and agent classes.

    Coordinate convention
    ---------------------
    Positions are integer voxel indices (x, y, z) with
      0 <= x < X,  0 <= y < Y,  0 <= z < Z.
    Observations and state vectors contain *normalized* coordinates divided by
    (X-1, Y-1, Z-1).
    """

    def __init__(
        self,
        args: Optional[argparse.Namespace] = None,
        num_uavs: int = 4,
        num_gbs: int = 4,
        grid_shape: Tuple[int, int, int] = (10, 10, 3),
        v_max: int = 2,
        T_max: int = 100,
        K_b: int = 10,
        delta_f: float = 180e3,
        P_tx_dBm: float = 23.0,
        beta_0: float = 1e-3,
        alpha: float = 2.0,
        sigma2_dBm: float = -104.0,
        beta_ema: float = 0.1,
        phi: float = 0.5,
        rho_col: float = 1.0,
        r_fail: float = 5.0,
        **kwargs,
    ) -> None:
        self.args = args

        # Allow args to override keyword defaults (enables --phi etc. from CLI)
        if args is not None:
            phi      = getattr(args, 'phi',      phi)
            rho_col  = getattr(args, 'rho_col',  rho_col)
            r_fail   = getattr(args, 'r_fail',   r_fail)
            v_max    = getattr(args, 'v_max',    v_max)
            T_max    = getattr(args, 'T_max',    T_max)
            K_b      = getattr(args, 'K_b',      K_b)
            num_uavs = getattr(args, 'num_uavs', num_uavs)

        # Grid
        self.X, self.Y, self.Z = grid_shape

        # Agents
        self.N = num_uavs
        self.M = num_gbs
        self.num_agents = num_uavs

        # Mobility
        self.v_max = v_max
        self.T_max = T_max
        self.episode_limit = T_max

        # Communication parameters
        self.K_b = K_b
        self.delta_f = delta_f
        self.P_tx = 10 ** ((P_tx_dBm - 30) / 10)      # dBm -> Watts
        self.sigma2 = 10 ** ((sigma2_dBm - 30) / 10)   # dBm -> Watts
        self.beta_0 = beta_0
        self.alpha = alpha
        self.beta_ema = beta_ema

        # Reward shaping weights
        self.phi = phi
        self.rho_col = rho_col
        self.r_fail = r_fail

        # ---- Action set ------------------------------------------------
        # Index 0 is always (0, 0, 0) — stay.  Then all other displacements
        # with L1 norm <= v_max in lexicographic order.
        raw = [(0, 0, 0)]
        for dx in range(-v_max, v_max + 1):
            for dy in range(-v_max, v_max + 1):
                for dz in range(-v_max, v_max + 1):
                    if (dx, dy, dz) != (0, 0, 0) and abs(dx) + abs(dy) + abs(dz) <= v_max:
                        raw.append((dx, dy, dz))
        self.action_set = np.array(raw, dtype=np.int32)  # (n_actions, 3)
        self.n_actions = len(self.action_set)             # 25 for v_max=2
        self.action_dim = self.n_actions

        # ---- GBS positions ---------------------------------------------
        # M=4: four corners of the XY plane at z=0
        corners = np.array([
            [0,          0,          0],
            [self.X - 1, 0,          0],
            [0,          self.Y - 1, 0],
            [self.X - 1, self.Y - 1, 0],
        ], dtype=np.float32)
        self.gbs_positions = corners[:self.M]  # (M, 3)

        # ---- Normalization constants -----------------------------------
        # Position: divide by (X-1, Y-1, Z-1) so coords -> [0, 1]
        self.pos_norm = np.array(
            [max(self.X - 1, 1), max(self.Y - 1, 1), max(self.Z - 1, 1)],
            dtype=np.float32,
        )
        self.gbs_positions_norm = self.gbs_positions / self.pos_norm  # (M, 3)

        # Rate: normalize by theoretical max (single UAV at 1-voxel distance)
        _snr_ref = self.P_tx * self.beta_0 / (self.sigma2 * 1.0 ** self.alpha)
        self.R_norm = float(self.K_b * self.delta_f * math.log2(1.0 + _snr_ref))

        # ---- Dimension bookkeeping -------------------------------------
        # obs:   q_i(3) + d_i(3) + T_min_i(1) + R_bar_i(1) + n_b(M) + g_b(M*3)
        self.obs_dim = 3 + 3 + 1 + 1 + self.M + self.M * 3
        # state: q*N(3N) + d*N(3N) + T_min*N(N) + R_bar*N(N) + n_b(M) + g_b(M*3)
        self.state_dim = 3 * self.N + 3 * self.N + self.N + self.N + self.M + self.M * 3

        self.obs_shape = (self.obs_dim,)
        self.state_shape = (self.state_dim,)
        self.action_shape = (self.action_dim,)
        self.reward_shape = (1,)
        self.done_shape = (1,)

        # ---- Episode state (set in reset) ------------------------------
        self.positions: np.ndarray = None       # (N, 3) int32
        self.origins: np.ndarray = None         # (N, 3) int32
        self.destinations: np.ndarray = None    # (N, 3) int32
        self.T_min: np.ndarray = None           # (N,) int32
        self.R_bar: np.ndarray = None           # (N,) float32 EMA rate
        self.arrived: np.ndarray = None         # (N,) bool
        self.arrival_step: np.ndarray = None    # (N,) int32
        self.last_actions: np.ndarray = None    # (N,) int32
        self._step: int = 0
        self._episode_score: float = 0.0
        self._all_arrived: bool = False

    # ==================================================================
    # Core MDP interface
    # ==================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Reset episode: randomize origins/destinations, zero EMA rates.

        Returns
        -------
        obs   : (N, actor_input_dim) float32
        state : (state_dim,)         float32
        info  : dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample unique origins and destinations.
        # Constraint: L1(origin_i, dest_i) >= 2 (not same cell or adjacent).
        # No two UAVs share origin; no two share destination; origins and
        # destinations are all distinct from each other.
        all_cells = [
            (x, y, z)
            for x in range(self.X)
            for y in range(self.Y)
            for z in range(self.Z)
        ]
        used: set = set()
        origins, destinations = [], []

        for _ in range(self.N):
            avail_origins = [c for c in all_cells if c not in used]
            origin = avail_origins[np.random.randint(len(avail_origins))]
            used.add(origin)

            avail_dests = [
                c for c in all_cells
                if c not in used
                and sum(abs(c[k] - origin[k]) for k in range(3)) >= 2
            ]
            dest = avail_dests[np.random.randint(len(avail_dests))]
            used.add(dest)

            origins.append(origin)
            destinations.append(dest)

        self.origins = np.array(origins, dtype=np.int32)
        self.destinations = np.array(destinations, dtype=np.int32)
        self.positions = self.origins.copy()

        # T_min_i = ceil(L1(origin_i, dest_i) / v_max)
        l1 = np.sum(np.abs(self.origins - self.destinations), axis=1)
        self.T_min = np.ceil(l1 / self.v_max).astype(np.int32)

        # Reset episode bookkeeping
        self.R_bar = np.zeros(self.N, dtype=np.float32)
        self.arrived = np.zeros(self.N, dtype=bool)
        self.arrival_step = np.full(self.N, self.T_max, dtype=np.int32)
        self.last_actions = np.zeros(self.N, dtype=np.int32)
        self._step = 0
        self._episode_score = 0.0
        self._all_arrived = False
        # Trajectory log: list of (N, 3) int32 position snapshots.
        # Index 0 = origins (before first step); index t = positions after step t.
        self._trajectory: List[np.ndarray] = [self.origins.copy()]

        info = {'episode_step': 0, 'episode_score': 0.0}
        obs = self._build_obs()
        state = self._build_state()
        obs = self.build_actor_inputs(
            obs, state, self.get_actions_one_hot(), self.get_agents_id_one_hot()
        )
        return np.array(obs, dtype=np.float32), np.array(state, dtype=np.float32), info

    def step(
        self, actions: Union[np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, dict]:
        """Apply one step for all UAVs.

        Parameters
        ----------
        actions : (N,) integer indices into self.action_set

        Returns
        -------
        obs        : (N, actor_input_dim) float32
        state      : (state_dim,)         float32
        reward     : scalar float32  (mean of per-agent rewards)
        terminated : bool  — all UAVs reached destination
        truncated  : bool  — T_max reached without full arrival
        info       : dict  — includes per-agent diagnostics
        """
        actions = np.asarray(actions, dtype=np.int32)

        prev_positions = self.positions.copy()
        prev_R_bar = self.R_bar.copy()

        # Move non-arrived UAVs; clip to grid bounds
        for i in range(self.N):
            if not self.arrived[i]:
                new_pos = self.positions[i] + self.action_set[actions[i]]
                new_pos = np.clip(
                    new_pos,
                    [0, 0, 0],
                    [self.X - 1, self.Y - 1, self.Z - 1],
                )
                self.positions[i] = new_pos

        self._step += 1
        self.last_actions = actions.copy()

        # Check arrival
        just_arrived = np.zeros(self.N, dtype=bool)
        for i in range(self.N):
            if not self.arrived[i] and np.array_equal(self.positions[i], self.destinations[i]):
                self.arrived[i] = True
                self.arrival_step[i] = self._step
                just_arrived[i] = True

        # EMA rate update
        rates = self._compute_rates()
        self.R_bar = (1.0 - self.beta_ema) * self.R_bar + self.beta_ema * rates

        # Record positions after move for trajectory logging
        self._trajectory.append(self.positions.copy())

        # Per-agent rewards -> mean team reward
        per_agent_rewards = self._compute_rewards(prev_positions, prev_R_bar, just_arrived)
        reward = float(np.mean(per_agent_rewards))

        # Termination flags
        self._all_arrived = bool(np.all(self.arrived))
        terminated = self._all_arrived
        truncated = (self._step >= self.T_max) and not terminated
        self._episode_score += reward

        info = {
            'episode_step': self._step,
            'episode_score': self._episode_score,
            'per_agent_rewards': per_agent_rewards,
            'rates': rates.copy(),
            'R_bar': self.R_bar.copy(),
            'arrived': self.arrived.copy(),
            'arrival_step': self.arrival_step.copy(),
        }

        obs = self._build_obs()
        state = self._build_state()
        obs = self.build_actor_inputs(
            obs, state, self.get_actions_one_hot(), self.get_agents_id_one_hot()
        )
        return (
            np.array(obs, dtype=np.float32),
            np.array(state, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Internal builders
    # ==================================================================

    def _build_obs(self) -> np.ndarray:
        """Build local observation vectors. Returns (N, obs_dim) float32."""
        n_b_norm = self._compute_n_b() / self.N          # (M,) in [0,1]
        gbs_flat = self.gbs_positions_norm.flatten()      # (M*3,)

        obs_list = []
        for i in range(self.N):
            q_norm = self.positions[i].astype(np.float32) / self.pos_norm
            d_norm = self.destinations[i].astype(np.float32) / self.pos_norm
            t_min_norm = float(self.T_min[i]) / float(self.T_max)
            r_bar_norm = float(self.R_bar[i]) / self.R_norm
            o_i = np.concatenate([
                q_norm,
                d_norm,
                [t_min_norm],
                [r_bar_norm],
                n_b_norm,
                gbs_flat,
            ]).astype(np.float32)
            obs_list.append(o_i)
        return np.stack(obs_list, axis=0)  # (N, obs_dim)

    def _build_state(self) -> np.ndarray:
        """Build global state vector. Returns (state_dim,) float32."""
        n_b_norm = self._compute_n_b() / self.N
        gbs_flat = self.gbs_positions_norm.flatten()

        q_all = (self.positions.astype(np.float32) / self.pos_norm).flatten()
        d_all = (self.destinations.astype(np.float32) / self.pos_norm).flatten()
        t_min_all = self.T_min.astype(np.float32) / float(self.T_max)
        r_bar_all = self.R_bar / self.R_norm

        return np.concatenate([
            q_all, d_all, t_min_all, r_bar_all, n_b_norm, gbs_flat,
        ]).astype(np.float32)

    def _get_gbs_assignment(self) -> np.ndarray:
        """Voronoi: assign each UAV to the nearest GBS. Returns (N,) int."""
        assignment = np.zeros(self.N, dtype=np.int32)
        pos_f = self.positions.astype(np.float32)   # (N, 3)
        for i in range(self.N):
            dists = np.linalg.norm(pos_f[i] - self.gbs_positions, axis=1)
            assignment[i] = int(np.argmin(dists))
        return assignment

    def _compute_n_b(self) -> np.ndarray:
        """Count UAVs per GBS. Returns (M,) float32."""
        assignment = self._get_gbs_assignment()
        n_b = np.zeros(self.M, dtype=np.float32)
        for b in assignment:
            n_b[b] += 1.0
        return n_b

    def _compute_rates(self) -> np.ndarray:
        """Compute instantaneous uplink rate for every UAV. Returns (N,) float32.

        R_i = kappa_i * delta_f * log2(1 + P_tx * beta_0 / (sigma^2 * dist^alpha))
        kappa_i = floor(K_b / n_b)   where n_b = |{j : GBS(j) == GBS(i)}|
        """
        assignment = self._get_gbs_assignment()
        n_b = np.zeros(self.M, dtype=np.int32)
        for b in assignment:
            n_b[b] += 1

        rates = np.zeros(self.N, dtype=np.float32)
        pos_f = self.positions.astype(np.float64)
        gbs_f = self.gbs_positions.astype(np.float64)

        for i in range(self.N):
            b = assignment[i]
            kappa = max(1, self.K_b // n_b[b])
            dist = float(np.linalg.norm(pos_f[i] - gbs_f[b])) + 1e-6
            snr = self.P_tx * self.beta_0 / (self.sigma2 * dist ** self.alpha)
            rates[i] = float(kappa * self.delta_f * math.log2(1.0 + snr))

        return rates

    def _compute_rewards(
        self,
        prev_positions: np.ndarray,
        prev_R_bar: np.ndarray,
        just_arrived: np.ndarray,
    ) -> np.ndarray:
        """Compute per-agent reward components. Returns (N,) float32.

        r_i = r_rate + r_nav + r_col + r_term
        """
        rewards = np.zeros(self.N, dtype=np.float32)

        for i in range(self.N):
            # Rate increment (EMA), normalized to [0, ~1] range
            r_rate = float(self.R_bar[i] - prev_R_bar[i]) / self.R_norm

            # Navigation potential shaping
            dist_before = float(np.linalg.norm(
                prev_positions[i].astype(float) - self.destinations[i].astype(float)
            ))
            dist_after = float(np.linalg.norm(
                self.positions[i].astype(float) - self.destinations[i].astype(float)
            ))
            r_nav = self.phi * (dist_before - dist_after)

            # Collision penalty (any other UAV at same voxel after move)
            r_col = 0.0
            for j in range(self.N):
                if j != i and np.array_equal(self.positions[i], self.positions[j]):
                    r_col = -self.rho_col
                    break

            # Terminal bonus / failure penalty (applied once).
            # R_bar normalized so the bonus is in a comparable scale to r_nav.
            r_term = 0.0
            if just_arrived[i]:
                T_i = float(self.arrival_step[i])
                r_term = (float(self.R_bar[i]) / self.R_norm) * float(self.T_min[i]) / T_i
            elif self._step >= self.T_max and not self.arrived[i]:
                r_term = -self.r_fail

            rewards[i] = r_rate + r_nav + r_col + r_term

        return rewards

    # ==================================================================
    # MARL observation / action interface
    # ==================================================================

    def get_obs(self) -> np.ndarray:
        """Returns (N, obs_dim) float32."""
        return self._build_obs()

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """Returns (obs_dim,) float32 for one agent."""
        return self._build_obs()[agent_id]

    def get_obs_size(self) -> int:
        return self.obs_dim

    def get_state(self) -> np.ndarray:
        """Returns (state_dim,) float32."""
        return self._build_state()

    def get_state_size(self) -> int:
        return self.state_dim

    def get_available_actions(self) -> np.ndarray:
        """Returns (N, n_actions) float32 binary mask.

        An action is available if the resulting position stays within the grid.
        Arrived UAVs may only stay (action index 0).
        """
        mask = np.zeros((self.N, self.n_actions), dtype=np.float32)
        for i in range(self.N):
            if self.arrived[i]:
                mask[i, 0] = 1.0
            else:
                for a in range(self.n_actions):
                    new_pos = self.positions[i] + self.action_set[a]
                    if (
                        0 <= new_pos[0] < self.X
                        and 0 <= new_pos[1] < self.Y
                        and 0 <= new_pos[2] < self.Z
                    ):
                        mask[i, a] = 1.0
        return mask

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        """Returns (n_actions,) mask for one agent."""
        return self.get_available_actions()[agent_id]

    def get_actions_one_hot(
        self, actions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns (N, n_actions) one-hot encoding of last (or given) actions."""
        if actions is None:
            actions = (
                self.last_actions
                if self.last_actions is not None
                else np.zeros(self.N, dtype=np.int32)
            )
        result = np.zeros((self.N, self.n_actions), dtype=np.float32)
        for i, a in enumerate(actions):
            result[i, int(a)] = 1.0
        return result

    def get_agents_id_one_hot(self) -> np.ndarray:
        """Returns (N, N) identity matrix as agent ID encodings."""
        return np.eye(self.N, dtype=np.float32)

    # ==================================================================
    # Actor input augmentation (mirrors SMACWrapperEnv)
    # ==================================================================

    def get_actor_input_dim(self) -> int:
        dim = self.obs_dim
        if self.args is not None:
            if getattr(self.args, 'use_global_state', False):
                dim += self.state_dim
            if getattr(self.args, 'use_last_actions', False):
                dim += self.n_actions
            if getattr(self.args, 'use_agents_id_onehot', False):
                dim += self.num_agents
        return dim

    def get_actor_input_shape(self) -> Tuple[int]:
        return (self.get_actor_input_dim(),)

    def build_actor_inputs(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        last_actions: np.ndarray,
        agents_id_onehot: np.ndarray,
    ) -> np.ndarray:
        """Optionally augment (N, obs_dim) with state / last-actions / agent-id."""
        inputs = obs  # (N, obs_dim)
        if self.args is not None:
            if getattr(self.args, 'use_global_state', False):
                state_exp = np.tile(state[None, :], (self.N, 1))
                inputs = np.concatenate([inputs, state_exp], axis=-1)
            if getattr(self.args, 'use_last_actions', False):
                inputs = np.concatenate([inputs, last_actions], axis=-1)
            if getattr(self.args, 'use_agents_id_onehot', False):
                inputs = np.concatenate([inputs, agents_id_onehot], axis=-1)
        return inputs

    # ==================================================================
    # Environment metadata
    # ==================================================================

    def get_env_info(self) -> dict:
        return {
            'obs_shape': self.obs_shape,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions,
            'actions_shape': self.n_actions,
            'num_agents': self.num_agents,
            'episode_limit': self.episode_limit,
        }

    @property
    def arrived_rate(self) -> float:
        """Fraction of UAVs that have reached their destination (0..1)."""
        return float(np.mean(self.arrived)) if self.arrived is not None else 0.0

    @property
    def win_counted(self) -> float:
        """Fraction of UAVs that arrived (0..1). Used as 'win_rate' in loggers."""
        return self.arrived_rate

    def get_trajectory(self) -> np.ndarray:
        """Return position history for the current episode.

        Returns
        -------
        np.ndarray, shape (T+1, N, 3), dtype int32
            Index 0 is origins; index t is positions after step t.
        """
        return np.stack(self._trajectory, axis=0)

    # ==================================================================
    # Gymnasium stubs (required by some runners)
    # ==================================================================

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def save_replay(self) -> None:
        pass
