from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Baselines
# ----------------------------

class BaselineAdapter(nn.Module):
    """Token-wise MLP + pooling (non-sequential baseline)."""
    def __init__(self, num_tokens: int = 60, d_model: int = 64, mlp_layers: int = 2, pool: str = "mean"):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.pool = pool

        layers = []
        for _ in range(mlp_layers):
            layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)  # [B,T,D]
        x = self.mlp(x)
        if self.pool == "mean":
            v = x.mean(dim=1)
        elif self.pool == "last":
            v = x[:, -1]
        else:
            raise ValueError(f"unknown pool={self.pool}")
        logits = self.head(v)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


class GRUBaseline(nn.Module):
    """Embedding -> GRU -> last hidden -> Linear."""
    def __init__(self, num_tokens: int = 60, d_model: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        _, h = self.gru(x)
        logits = self.head(h[-1])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


# ----------------------------
# Executors / Plugins
# ----------------------------

class A5ExactScan(nn.Module):
    """
    Fixed executor using Cayley table.
    State s is group id (Long).
    """
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.long()
        self.register_buffer("mul", mul_table)  # [60,60]
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                no_scan: bool = False, shuffle_M: bool = False, reset_each_step: bool = False):
        B, T = input_ids.shape
        device = input_ids.device
        x = input_ids
        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=device)
            x = x[:, perm]

        if no_scan:
            s = x[:, -1]
        else:
            s = torch.full((B,), self.id_id, device=device, dtype=torch.long)
            for t in range(T):
                if reset_each_step:
                    s = torch.full((B,), self.id_id, device=device, dtype=torch.long)
                s = self.mul[x[:, t], s]

        logits = torch.zeros((B, self.num_tokens), device=device)
        logits.scatter_(1, s.view(-1, 1), 5.0)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss


class Route1SoftScan(nn.Module):
    """
    route1: learn token -> distribution over 60 actions (group elements),
    then execute via *soft state distribution* scan using Cayley table.

    loss = final_loss + aux_weight * route_loss
    - final_loss: CE on final state distribution (log)
    - route_loss: CE(logits_g, token) makes router learn identity mapping (bootstrap)
      You can anneal aux_weight with _aux_weight_override from trainer.
    """
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60, temp: float = 1.0, aux_weight: float = 5.0):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.long()
        self.register_buffer("mul", mul_table)
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)
        self.temp = float(temp)

        self.route_logits = nn.Parameter(torch.zeros(num_tokens, num_tokens))
        self.loss_fn = nn.CrossEntropyLoss()

        self.aux_weight = float(aux_weight)
        self._aux_weight_override = None  # trainer may set

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                no_scan: bool = False, shuffle_M: bool = False, reset_each_step: bool = False):
        B, T = input_ids.shape
        device = input_ids.device

        x = input_ids
        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=device)
            x = x[:, perm]

        logits_g = self.route_logits[x]  # [B,T,60]
        probs_g = torch.softmax(logits_g / max(self.temp, 1e-6), dim=-1)

        # soft scan over state distribution s_dist: [B,60]
        if no_scan:
            s_dist = probs_g[:, -1]
        else:
            s_dist = torch.zeros(B, self.num_tokens, device=device)
            s_dist[:, self.id_id] = 1.0
            for t in range(T):
                if reset_each_step:
                    s_dist.zero_()
                    s_dist[:, self.id_id] = 1.0

                g_dist = probs_g[:, t]  # [B,60]
                next_s = torch.zeros_like(s_dist)
                # next_s[h] += sum_s s_dist[s] * g_dist[g] where h = mul[g,s]
                # implement by looping g (60 is small; fine)
                for g in range(self.num_tokens):
                    h = self.mul[g]  # [60]
                    next_s.index_add_(1, h, s_dist * g_dist[:, g].unsqueeze(1))
                s_dist = next_s

        logits_final = (s_dist.clamp_min(1e-9)).log()

        loss = None
        if labels is not None:
            loss_final = self.loss_fn(logits_final, labels)
            loss_route = F.cross_entropy(logits_g.reshape(-1, self.num_tokens), x.reshape(-1))

            w = self.aux_weight
            if self._aux_weight_override is not None:
                w = float(self._aux_weight_override)
            loss = loss_final + w * loss_route

        return logits_final, loss
