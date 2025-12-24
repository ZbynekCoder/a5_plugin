import torch
import torch.nn as nn


class PairMLPTagger(nn.Module):
    """
    Sanity baseline for A5 multiplication on length-2 sequences.
    Learns y1 = g2 ∘ g1 (60-way classification).
    """
    def __init__(self, num_tokens: int = 60, hidden: int = 128, mlp_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_tokens),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        """
        input_ids: [B, 2]
        labels:    [B, 2] (we only use labels[:, 1])
        """
        assert input_ids.dim() == 2 and input_ids.size(1) == 2, "PairMLP expects sequences of length 2."
        a = input_ids[:, 0]
        b = input_ids[:, 1]

        ea = self.embed(a)
        eb = self.embed(b)
        x = torch.cat([ea, eb], dim=-1)  # [B, 2H]
        logits = self.mlp(x)            # [B, 60]

        loss = None
        if labels is not None:
            y = labels[:, 1].contiguous()   # only second position
            loss = self.loss_fn(logits, y)

        return logits, loss
