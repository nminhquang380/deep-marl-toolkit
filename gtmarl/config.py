"""Configuration for the GT-MARL LAWN experiments (TWC manuscript v18).

Equation numbers refer to the manuscript. All distances used by the mobility
model (d_safe, d_comm, v_max) are in voxel units; the radio model uses metres.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    # ---- Airspace (Sec. II-B) -------------------------------------------
    W: int = 32                 # voxels along x
    R: int = 32                 # voxels along y
    H: int = 3                  # altitude layers
    ds: float = 40.0            # voxel resolution Delta_s [m]
    h_min: float = 60.0         # altitude of the lowest layer centre [m]

    # ---- Network (Sec. II) -------------------------------------------------
    N: int = 20                 # number of UAVs
    M: int = 4                  # number of GBSs (square grid: 4, 9, 16, ...)
    gbs_xy: Optional[List[Tuple[float, float]]] = None  # voxel coords; None -> regular grid
    gbs_height: float = 25.0    # [m]
    K_b: int = 22               # RBs per GBS (Eq. (6)); ~66% mean load at N=20
    K_b_scale_with_N: bool = False  # K_b <- ceil(K_b * N / N_ref) keeps the load factor
    N_ref: int = 20

    # ---- Radio (Eqs. (2)-(4)) ----------------------------------------------
    delta_f: float = 180e3      # RB bandwidth [Hz]
    P_tx_dBm: float = 0.0       # effective per-RB power toward the UAV incl. sidelobe loss [dBm]
    beta0_dB: float = -43.3     # reference channel gain at 1 m (3.5 GHz free space)
    alpha: float = 2.3          # path-loss exponent (LoS-dominant A2G)
    noise_dBm: float = -112.4   # per-RB noise power (-174 + 10log10(180e3) + 9 dB NF)
    snr_th_dB: Tuple[float, ...] = (24.0, 16.0, 11.0, 6.0)  # gamma_bar^(1) > ... > gamma_bar^(L)
    R_min: float = 1.4e6        # GBR of the C2 flow [bit/s]
    fading: str = "none"        # "none" (LoS, v18) | "rician" (path loss + small-scale fading)
    rician_K_dB: float = 10.0

    # ---- Mobility (Eqs. (8)-(10), (17)) -------------------------------------
    v_max: int = 1
    d_safe: float = 1.0
    d_comm: float = 4.0         # must satisfy d_comm >= d_safe + 2 v_max
    eta: float = 1.5            # deadline tolerance, H_i = floor(eta * T_i^min)
    min_od_l1: int = 8          # minimum origin-destination L1 distance [voxels]

    # ---- Cost / reward (Eqs. (19), (24)) -----------------------------------
    mu: float = 2.0             # penalty weight (Prop. 2 suggests mu > eta)
    rho_fail: float = 10.0
    w_fail_dist: float = 0.1    # graded failure cost rho_fail + w * ||q_i(H_i) - d_i|| (0 = v18 Eq. (19))
    rho_nav: float = 0.05

    # ---- Episode generation ------------------------------------------------
    init_no_overload: bool = True   # operator-controlled initial configuration

    def effective_K(self) -> int:
        if self.K_b_scale_with_N:
            import math
            return int(math.ceil(self.K_b * self.N / self.N_ref))
        return self.K_b


@dataclass
class TrainConfig:
    algo: str = "gippo"         # gippo | ippo | mappo
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
    if isinstance(target, tuple):
        return tuple(float(v) for v in value.strip("()[] ").split(",") if v)
    if target is None:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
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
    if hasattr(obj, "snr_th_dB"):
        obj.snr_th_dB = tuple(obj.snr_th_dB)
    return obj
