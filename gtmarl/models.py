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


class EGAT(nn.Module):
    """Ego-centric edge-featured graph attention encoder (Sec. IV-B).

    The local graph of UAV i has nodes {i} U M_i (the nbr_cap nearest beacons) and an edge
    (j, k) whenever ||q_j - q_k|| <= d_comm, so the neighbours are also related to each other.
    Edge features e_jk = [q_k - q_j, ||q_k - q_j||, v_k - v_j, 1(b_k = b_j)] are computed from
    the beacons alone. Each layer is a multi-head graph attention with edge features,

        alpha_jk = softmax_k LeakyReLU(a_s^T W h_j + a_t^T W h_k + a_e^T W_e e_jk),
        h_j <- LayerNorm(h_j + ELU(W_o [sum_k alpha_jk W h_k]_heads)),

    and the ego node's final state is read out. Requires obs_version 2.
    """

    EDGE_DIM = 8

    def __init__(self, own_dim: int, nbr_dim: int, hidden: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.H, self.L, self.dh = heads, hidden, layers, hidden // heads
        self.inp_ego = nn.Linear(own_dim, hidden)
        self.inp_nbr = nn.Linear(nbr_dim, hidden)
        self.W = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.We = nn.ModuleList([nn.Linear(self.EDGE_DIM, hidden, bias=False) for _ in range(layers)])
        self.a_s = nn.ParameterList([nn.Parameter(torch.randn(heads, self.dh) * 0.1) for _ in range(layers)])
        self.a_t = nn.ParameterList([nn.Parameter(torch.randn(heads, self.dh) * 0.1) for _ in range(layers)])
        self.a_e = nn.ParameterList([nn.Parameter(torch.randn(heads, self.dh) * 0.1) for _ in range(layers)])
        self.Wo = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, own: torch.Tensor, nb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # own (..., own_dim); nb (..., C, nbr_dim); valid (..., C) in {0, 1}
        lead = own.shape[:-1]
        own = own.reshape(-1, own.shape[-1]); nb = nb.reshape(-1, *nb.shape[-2:])
        valid = valid.reshape(-1, valid.shape[-1])
        Bn, n, h, dh = own.shape[0], nb.shape[1] + 1, self.h, self.dh
        z = torch.zeros_like(own[:, :3])
        P = torch.cat([z[:, None], nb[..., 0:3]], 1)                        # (q_j - q_i) / d_comm
        V = torch.cat([own[:, None, 12:15], nb[..., 6:9]], 1)               # velocities
        G = torch.cat([own[:, None, 3:6], nb[..., 9:12]], 1)                # g_{b_j} - q_i
        ok = torch.cat([torch.ones_like(valid[:, :1]), valid], 1) > 0.5     # (B, n)
        dP = P[:, None] - P[:, :, None]                                     # [j, k] = P_k - P_j
        dist = dP.norm(dim=-1, keepdim=True)
        same = ((G[:, None] - G[:, :, None]).abs().sum(-1, keepdim=True) < 1e-4).float()
        e = torch.cat([dP, dist, V[:, None] - V[:, :, None], same], -1)     # (B, n, n, 8)
        adj = ok[:, :, None] & ok[:, None, :] & (dist[..., 0] <= 1.0 + 1e-6)
        neg = (~adj).unsqueeze(-1) * -1e9
        x = torch.cat([self.inp_ego(own)[:, None], self.inp_nbr(nb)], 1)    # (B, n, H)
        okf = ok.unsqueeze(-1).float()
        for l in range(self.L):
            Wx = self.W[l](x).view(Bn, n, h, dh)
            We = self.We[l].weight.view(h, dh, self.EDGE_DIM)                # per-head edge map
            ue = (self.a_e[l][:, :, None] * We).sum(1)                       # a_e^T W_e, (h, 8)
            src = (Wx * self.a_s[l]).sum(-1)                                 # (B, n, h)
            tgt = (Wx * self.a_t[l]).sum(-1)
            score = src[:, :, None] + tgt[:, None] + e @ ue.t()              # (B, n, n, h)
            alpha = torch.softmax(nn.functional.leaky_relu(score, 0.2) + neg, dim=2)
            msg = torch.einsum("bjkh,bkhd->bjhd", alpha, Wx).reshape(Bn, n, self.H)   # sum_k a W h_k
            x = self.norm[l](x + nn.functional.elu(self.Wo[l](msg))) * okf
        return x[:, 0].reshape(*lead, self.H)                                # ego node


class ActorCritic(nn.Module):
    """Shared encoder f_omega, actor pi_theta and critic V_psi (Eqs. (29)-(31)).

    nbr = "egat"  -> e_i = f_omega([nu_i, h_i]) with h_i the ego state of the EGAT encoder
                     (proposed; obs_version 2)
    nbr = "graph" -> e_i = f_omega([nu_i, xi_bar_i, |M_i|])        (v22 Eq. (29)-(30), mean)
    nbr = "graph_msg" -> e_i = f_omega([nu_i, mean_j phi(xi_ji), |M_i|]) with a message
                     MLP phi applied to every neighbour before the mean (GraphSAGE-mean)
    nbr = "knn"   -> e_i = f_omega([nu_i, xi_(1)i, ..., xi_(k)i])  (IPPO-MLP benchmark)
    nbr = "none"  -> e_i = f_omega(nu_i)                           (no neighbour information)
    state_dim > 0 -> centralised critic V(s, o_i) on the joint state plus the agent's own
                     observation; the encoder is then trained by the actor (and aux) loss only.
    aux = True    -> congestion head predicting the normalised demands D_b(t+1) / K_b.
    """

    def __init__(self, n_gbs: int, n_actions: int, hidden: int = 128, emb_dim: int = 64,
                 nbr: str = "graph", knn_k: int = 4, state_dim: int = 0, obs_dim: int = 0,
                 nbr_cap: int = 8, msg_dim: int = 32, own_dim: int = OWN_DIM, nbr_dim: int = NBR_DIM,
                 aux: bool = False, heads: int = 4, gnn_layers: int = 2):
        super().__init__()
        self.kw = dict(n_gbs=n_gbs, n_actions=n_actions, hidden=hidden, emb_dim=emb_dim, nbr=nbr,
                       knn_k=knn_k, state_dim=state_dim, obs_dim=obs_dim, nbr_cap=nbr_cap,
                       msg_dim=msg_dim, own_dim=own_dim, nbr_dim=nbr_dim, aux=aux, heads=heads,
                       gnn_layers=gnn_layers)
        self.nbr, self.knn_k, self.n_gbs, self.nbr_cap = nbr, knn_k, n_gbs, max(nbr_cap, knn_k)
        self.own_dim, self.nbr_dim = own_dim, nbr_dim
        self.use_graph = nbr in ("graph", "graph_msg", "egat")
        if nbr == "graph_msg":
            self.phi = nn.Sequential(mlp(nbr_dim, 64, msg_dim, n_hidden=1), nn.Tanh())
        if nbr == "egat":
            assert own_dim >= 15 and nbr_dim >= 12, "egat needs obs_version 2"
            self.gnn = EGAT(own_dim, nbr_dim, hidden=emb_dim, heads=heads, layers=gnn_layers)
        enc_in = own_dim + {"graph": nbr_dim + 1, "graph_msg": msg_dim + 1, "knn": nbr_dim * knn_k,
                            "none": 0, "egat": emb_dim}[nbr]
        self.encoder = nn.Sequential(mlp(enc_in, hidden, emb_dim, n_hidden=1), nn.Tanh())
        z_dim = emb_dim + n_gbs
        self.actor = mlp(z_dim, hidden, n_actions, out_gain=0.01)
        self.central = state_dim > 0
        if self.central:
            self.critic = mlp(state_dim + obs_dim, 2 * hidden, 1)
        else:
            self.critic = mlp(z_dim, hidden, 1)
        self.aux = aux
        if aux:
            self.aux_head = mlp(z_dim, hidden, n_gbs, n_hidden=1)
        self.vnorm = ValueNorm()

    def _z(self, obs: torch.Tensor) -> torch.Tensor:
        own = obs[..., :self.own_dim]
        o = self.own_dim
        graph = obs[..., o:o + self.nbr_dim + 1]
        o += self.nbr_dim + 1
        block = obs[..., o:o + self.nbr_dim * self.nbr_cap]
        dem = obs[..., -self.n_gbs:]
        nb = block.reshape(*block.shape[:-1], self.nbr_cap, self.nbr_dim)
        valid = (nb[..., 3] > 0).float()                                   # normalised distance > 0
        if self.nbr == "graph":
            x = torch.cat([own, graph], -1)
        elif self.nbr == "graph_msg":
            msg = (self.phi(nb) * valid[..., None]).sum(-2) / valid.sum(-1, keepdim=True).clamp(min=1.0)
            x = torch.cat([own, msg, graph[..., -1:]], -1)                  # graph[..., -1] = |M_i| / 8
        elif self.nbr == "egat":
            x = torch.cat([own, self.gnn(own, nb, valid)], -1)
        elif self.nbr == "knn":
            x = torch.cat([own, block[..., :self.nbr_dim * self.knn_k]], -1)
        else:
            x = own
        return torch.cat([self.encoder(x), dem], -1)

    def predict_demand(self, obs: torch.Tensor) -> torch.Tensor:
        return self.aux_head(self._z(obs))

    def forward_all(self, obs: torch.Tensor, mask: torch.Tensor, state: torch.Tensor = None,
                    with_value: bool = True):
        """Policy, normalised value and z from a single pass of the encoder."""
        z = self._z(obs)
        d = torch.distributions.Categorical(logits=self.actor(z).masked_fill(~mask, -1e9))
        if not with_value:                 # execution: the critic is not used (and a centralised
            return d, torch.zeros(obs.shape[:-1]), z   # critic is tied to the training N)
        if self.central:
            st = state if state.dim() == obs.dim() else \
                state.unsqueeze(-2).expand(*obs.shape[:-1], state.shape[-1])
            v = self.critic(torch.cat([st, obs], -1)).squeeze(-1)
        else:
            v = self.critic(z).squeeze(-1)
        return d, v, z

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
