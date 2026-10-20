"""Configuration for the GT-MARL LAWN experiments (TWC manuscript v22).

Equation numbers refer to manuscript v22. Mobility distances (d_safe, d_comm,
v_max) are in voxel units; the radio model uses metres. Code name ``M`` is the
number of GBSs (B in the manuscript); ``m`` / ``beta`` is the bearer size.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# SNR thresholds [dB] per number of levels L (Sec. V-E); the lowest threshold is
# the coverage level gamma_bar^(L), common to all L.
SNR_PRESETS = {
    1: (10.0,),                     # bearer 5 RBs
    2: (17.1, 10.0),                # 3 / 5
    3: (25.6, 17.1, 10.0),          # 2 / 3 / 5
    4: (25.6, 17.1, 12.8, 10.0),    # 2 / 3 / 4 / 5   (thresholds = ergodic-rate inverses + 0.1 dB)
}


@dataclass
class EnvConfig:
    # ---- Airspace (Sec. II-B, V-A) --------------------------------------------
    W: int = 60                 # voxels along x  (1.5 km / 25 m)
    R: int = 60                 # voxels along y
    H: int = 4                  # altitude layers in [50, 150] m
    ds: float = 25.0            # voxel resolution Delta_s [m]
    h_min: float = 62.5         # altitude of the lowest voxel centre [m]

    # ---- Network (Sec. II, V-A) ------------------------------------------------
    N: int = 30                 # number of UAVs
    M: int = 4                  # number of GBSs (B); square grid -> ISD 750 m for B = 4
    gbs_xy: Optional[List[Tuple[float, float]]] = None   # voxel coords; None -> regular grid
    gbs_height: float = 25.0    # [m]
    K_b: int = 51               # RBs per GBS (20 MHz carrier, 30 kHz SCS)
    K_b_scale_with_N: bool = False
    N_ref: int = 30

    # ---- Radio (Eqs. (2)-(9)) ------------------------------------------------
    delta_f: float = 360e3      # RB bandwidth (12 x 30 kHz) [Hz]
    P_tx_dBm: float = 6.5       # effective per-RB transmit power toward the UAV [dBm]
    G0_dB: float = -43.3        # reference channel gain at 1 m (3.5 GHz free space)
    alpha: float = 2.3          # path-loss exponent
    noise_dBm: float = -111.4   # per-RB noise (-174 + 10log10(360e3) + 7 dB NF)
    L: int = 3                  # number of SNR levels (thresholds from SNR_PRESETS)
    snr_th_dB: Optional[Tuple[float, ...]] = None        # overrides the preset if given
    R_min: float = 6e6          # GBR of the C2 flow [bit/s]
    fading: str = "rician"      # "rician" (ergodic rate, Eq. (4)) | "none" (log2(1+gamma))
    rician_K_dB: float = 10.0   # Rician factor vartheta

    # ---- Mobility (Eqs. (10)-(12), (19)) ---------------------------------------
    v_max: int = 1
    d_safe: float = 1.0
    d_comm: float = 4.0         # must satisfy d_comm >= d_safe + 2 v_max
    eta: float = 1.5
    min_od_l1: int = 12         # minimum origin-destination L1 distance [voxels]

    # ---- Origins: depots (Sec. V-A) -------------------------------------------
    origin_mode: str = "depots"  # "depots" | "uniform"
    depots: Tuple[Tuple[float, float], ...] = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75))  # area fractions
    depot_radius: float = 7.0   # [voxels]
    depot_layers: int = 2       # origins in the lowest layers

    # ---- Cost / reward (Eqs. (14), (21), (25)) ----------------------------------
    mu: float = 4.5             # penalty weight, 3 * eta (Remark 1)
    c_fail: float = 100.0      # must exceed the penalty a UAV can accumulate (mu * beta per slot)
    w_fail_dist: float = 0.5    # graded failure cost c_fail + w ||q_i - d_i|| (0 = v22 Eq. (21))
    w_nav: float = 0.05         # shaping weight (0 = ablation without shaping)
    penalty: str = "marginal"   # "marginal" (Eq. (14)) | "lost" (beta_i - kappa_i, ablation)
    ablate: Tuple[str, ...] = ()  # subset of {"cocell", "beta_nbr", "own_pos"} (Sec. V-F)
    knn_k: int = 4              # neighbours of the IPPO-MLP benchmark

    # ---- Episode generation -------------------------------------------------
    init_check: str = "soft"    # "strict": initial and final configurations overload no cell (Sec. II-A);
                                # "soft": strict if found within the retry budget, else origins only,
                                # else unconstrained (large N at B = 4); "off"

    def effective_K(self) -> int:
        if self.K_b_scale_with_N:
            return int(math.ceil(self.K_b * self.N / self.N_ref))
        return self.K_b

    def thresholds(self) -> Tuple[float, ...]:
        return tuple(self.snr_th_dB) if self.snr_th_dB else SNR_PRESETS[self.L]


@dataclass
class TrainConfig:
    algo: str = "gippo"         # gippo | ippo_mlp | ippo | mappo
    seed: int = 0
    iters: int = 1500
    n_envs: int = 32            # episodes per iteration (N_ep in Alg. 1)
    lr: float = 3e-4
    clip: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    epochs: int = 5
    minibatches: int = 4
    max_grad_norm: float = 0.5
    hidden: int = 128
    emb_dim: int = 64
    device: str = "cpu"
    threads: int = 1
    eval_every: int = 50
    eval_episodes: int = 64
    save_every: int = 100
    out: str = "runs"
    tag: str = ""


def _coerce(value: str, target):
    if isinstance(target, bool):
        return value.lower() in ("1", "true", "yes", "y")
    if isinstance(target, int) and not isinstance(target, bool):
        return int(value)
    if isinstance(target, float):
        return float(value)
    if isinstance(target, tuple) or target is None:
        try:
            v = json.loads(value)
            return tuple(v) if isinstance(v, list) else v
        except json.JSONDecodeError:
            items = [x for x in value.strip("()[] ").split(",") if x]
            try:
                return tuple(float(x) for x in items)
            except ValueError:
                return tuple(items)
    return value


def apply_overrides(cfg, pairs: List[str]):
    """Apply ``key=value`` overrides to a dataclass in place."""
    for p in pairs or []:
        k, v = p.split("=", 1)
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, _coerce(v, getattr(cfg, k)))
    return cfg


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(cls, d: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    obj = cls(**{k: v for k, v in d.items() if k in names})
    for k in ("snr_th_dB", "ablate"):
        if hasattr(obj, k) and isinstance(getattr(obj, k), list):
            setattr(obj, k, tuple(getattr(obj, k)))
    if hasattr(obj, "depots"):
        obj.depots = tuple(tuple(d) for d in obj.depots)
    return obj
