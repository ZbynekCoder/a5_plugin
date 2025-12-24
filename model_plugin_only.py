import torch
import torch.nn as nn


class BaselineAdapter(nn.Module):
    def __init__(
        self,
        num_tokens: int = 60,
        d_model: int = 64,
        mlp_layers: int = 2,
        pool: str = "mean",
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.pool = pool

        layers = []
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)  # [B,T,D]
        x = self.mlp(x)            # [B,T,D]

        if self.pool == "mean":
            v = x.mean(dim=1)
        elif self.pool == "last":
            v = x[:, -1]
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        logits = self.head(v)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss


class StateScanPlugin(nn.Module):
    def __init__(
        self,
        num_tokens: int = 60,
        d_model: int = 64,
        eps: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.repr = nn.Parameter(torch.randn(num_tokens, d_model, d_model) * 0.02)
        self.eps = eps  # 保留参数接口，当前不使用

        self.readout = nn.Sequential(
            nn.Linear(d_model * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_tokens),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        no_scan: bool = False,
        shuffle_M: bool = False,
        zero_state: bool = False,
    ):
        B, T = input_ids.shape
        d = self.d_model

        M = self.repr[input_ids]  # [B,T,d,d]

        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=input_ids.device)
            M = M[:, perm]

        M_bt = M.reshape(B * T, d, d)
        Q, _ = torch.linalg.qr(M_bt)
        M = Q.reshape(B, T, d, d)

        if no_scan:
            S = M[:, -1]
        else:
            S = torch.eye(d, device=input_ids.device).unsqueeze(0).repeat(B, 1, 1)
            for t in range(T):
                S = torch.bmm(M[:, t], S)

        if zero_state:
            S = torch.eye(d, device=input_ids.device).unsqueeze(0).repeat(B, 1, 1)

        v = S.reshape(B, -1)
        logits = self.readout(v)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss
