"""Policy and value networks (Sec. IV-B, Eqs. (26)-(30))."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import GRAPH_DIM, OWN_DIM


def mlp(inp: int, hidden: int, out: int, n_hidden: int = 2, out_gain: float = 1.0) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(n_hidden):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = hidden
    lin = nn.Linear(d, out)
    nn.init.orthogonal_(lin.weight, out_gain)
    nn.init.zeros_(lin.bias)
    layers.append(lin)
    return nn.Sequential(*layers)


class ValueNorm(nn.Module):
    """Running normalisation of value targets (debiased EMA, as in MAPPO)."""

    def __init__(self, beta: float = 0.995, eps: float = 1e-5):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("mean_sq", torch.zeros(()))
        self.register_buffer("debias", torch.zeros(()))

    def stats(self):
        d = self.debias.clamp(min=self.eps)
        mean = self.mean / d
        var = (self.mean_sq / d - mean ** 2).clamp(min=1e-2)
        return mean, var.sqrt()

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        self.mean.mul_(self.beta).add_((1 - self.beta) * x.mean())
        self.mean_sq.mul_(self.beta).add_((1 - self.beta) * (x ** 2).mean())
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, x):
        m, s = self.stats()
        return (x - m) / s

    def denormalize(self, x):
        m, s = self.stats()
        return x * s + m


class ActorCritic(nn.Module):
    """Shared encoder f_omega, actor pi_theta and critic V_psi.

    use_graph=True  -> h_i = f_omega([nu_i, xi_bar_i, |M_i|])  (proposed, Eq. (29))
    use_graph=False -> h_i = f_omega(nu_i)                      (ablation without the GNN)
    central_critic  -> V(s) on the joint state (MAPPO baseline); the encoder is then
                       trained by the actor loss only.
    """

    def __init__(self, n_gbs: int, n_actions: int, hidden: int = 128, emb_dim: int = 64,
                 use_graph: bool = True, state_dim: int = 0):
        super().__init__()
        self.use_graph = use_graph
        self.n_gbs = n_gbs
        enc_in = OWN_DIM + (GRAPH_DIM if use_graph else 0)
        self.encoder = nn.Sequential(mlp(enc_in, hidden, emb_dim, n_hidden=1), nn.Tanh())
        z_dim = emb_dim + n_gbs
        self.actor = mlp(z_dim, hidden, n_actions, out_gain=0.01)
        self.central = state_dim > 0
        if self.central:
            self.critic = mlp(state_dim, 2 * hidden, 1)
        else:
            self.critic = mlp(z_dim, hidden, 1)
        self.vnorm = ValueNorm()

    def _z(self, obs: torch.Tensor) -> torch.Tensor:
        own = obs[..., :OWN_DIM]
        graph = obs[..., OWN_DIM:OWN_DIM + GRAPH_DIM]
        dem = obs[..., OWN_DIM + GRAPH_DIM:]
        x = torch.cat([own, graph], -1) if self.use_graph else own
        return torch.cat([self.encoder(x), dem], -1)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.actor(self._z(obs))
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """Normalised value estimate; use ``self.vnorm.denormalize`` for returns."""
        if self.central:
            return self.critic(state).squeeze(-1)
        return self.critic(self._z(obs)).squeeze(-1)
