"""Policy and value networks (Sec. IV-B, Eqs. (26)-(30))."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import GRAPH_DIM, NBR_DIM, OWN_DIM


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
    """Shared encoder f_omega, actor pi_theta and critic V_psi (Eqs. (29)-(31)).

    nbr = "graph" -> e_i = f_omega([nu_i, xi_bar_i, |M_i|])        (v22 Eq. (29)-(30))
    nbr = "graph_msg" -> e_i = f_omega([nu_i, mean_j phi(xi_ji), |M_i|]) with a message
                     MLP phi applied to every neighbour before the mean (GraphSAGE-mean)
    nbr = "knn"   -> e_i = f_omega([nu_i, xi_(1)i, ..., xi_(k)i])  (IPPO-MLP benchmark)
    nbr = "none"  -> e_i = f_omega(nu_i)                           (no neighbour information)
    state_dim > 0 -> centralised critic V(s, o_i) on the joint state plus the agent's own
                     observation ("agent-specific global state" of MAPPO [Yu et al.]);
                     the encoder is then trained by the actor loss only.
    """

    def __init__(self, n_gbs: int, n_actions: int, hidden: int = 128, emb_dim: int = 64,
                 nbr: str = "graph", knn_k: int = 4, state_dim: int = 0, obs_dim: int = 0,
                 nbr_cap: int = 8, msg_dim: int = 32):
        super().__init__()
        self.nbr, self.knn_k, self.n_gbs, self.nbr_cap = nbr, knn_k, n_gbs, max(nbr_cap, knn_k)
        self.use_graph = nbr in ("graph", "graph_msg")
        if nbr == "graph_msg":
            self.phi = nn.Sequential(mlp(NBR_DIM, 64, msg_dim, n_hidden=1), nn.Tanh())
        enc_in = OWN_DIM + {"graph": GRAPH_DIM, "graph_msg": msg_dim + 1, "knn": NBR_DIM * knn_k,
                            "none": 0}[nbr]
        self.encoder = nn.Sequential(mlp(enc_in, hidden, emb_dim, n_hidden=1), nn.Tanh())
        z_dim = emb_dim + n_gbs
        self.actor = mlp(z_dim, hidden, n_actions, out_gain=0.01)
        self.central = state_dim > 0
        if self.central:
            self.critic = mlp(state_dim + obs_dim, 2 * hidden, 1)
        else:
            self.critic = mlp(z_dim, hidden, 1)
        self.vnorm = ValueNorm()

    def _z(self, obs: torch.Tensor) -> torch.Tensor:
        own = obs[..., :OWN_DIM]
        o = OWN_DIM
        graph = obs[..., o:o + GRAPH_DIM]
        o += GRAPH_DIM
        block = obs[..., o:o + NBR_DIM * self.nbr_cap]
        dem = obs[..., -self.n_gbs:]
        if self.nbr == "graph":
            x = torch.cat([own, graph], -1)
        elif self.nbr == "graph_msg":
            nb = block.reshape(*block.shape[:-1], self.nbr_cap, NBR_DIM)
            valid = (nb[..., 3] > 0).float()                         # normalised distance > 0
            msg = (self.phi(nb) * valid[..., None]).sum(-2) / valid.sum(-1, keepdim=True).clamp(min=1.0)
            x = torch.cat([own, msg, graph[..., -1:]], -1)          # graph[..., -1] = |M_i| / 8
        elif self.nbr == "knn":
            x = torch.cat([own, block[..., :NBR_DIM * self.knn_k]], -1)
        else:
            x = own
        return torch.cat([self.encoder(x), dem], -1)

    def dist(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.actor(self._z(obs))
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """Normalised value estimate; use ``self.vnorm.denormalize`` for returns."""
        if self.central:
            st = state if state.dim() == obs.dim() else \
                state.unsqueeze(-2).expand(*obs.shape[:-1], state.shape[-1])
            return self.critic(torch.cat([st, obs], -1)).squeeze(-1)
        return self.critic(self._z(obs)).squeeze(-1)
