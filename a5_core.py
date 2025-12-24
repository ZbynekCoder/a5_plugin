import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset


# ----------------------------
# Group: A5 generation
# ----------------------------

def _compose(p: Tuple[int, ...], q: Tuple[int, ...]) -> Tuple[int, ...]:
    """Permutation composition: (p ∘ q)(i) = p[q[i]]"""
    return tuple(p[q[i]] for i in range(len(p)))


def _parity(perm: Tuple[int, ...]) -> int:
    """Return 0 if even, 1 if odd."""
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            inv += perm[i] > perm[j]
    return inv % 2


def generate_a5() -> Tuple[List[Tuple[int, ...]], List[List[int]], int]:
    """
    Returns:
      elements: list of 60 even permutations of (0..4)
      mul: 60x60 Cayley table, mul[a][b] = a ∘ b (left multiply)
      id_id: index of identity
    """
    base = list(range(5))
    # generate all permutations of 5
    perms = []
    def backtrack(cur, used):
        if len(cur) == 5:
            perms.append(tuple(cur))
            return
        for i in range(5):
            if not used[i]:
                used[i] = True
                cur.append(i)
                backtrack(cur, used)
                cur.pop()
                used[i] = False
    backtrack([], [False] * 5)

    elements = [p for p in perms if _parity(p) == 0]  # A5
    assert len(elements) == 60

    idx = {p: i for i, p in enumerate(elements)}
    id_perm = tuple(base)
    id_id = idx[id_perm]

    mul = [[0] * 60 for _ in range(60)]
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            c = _compose(a, b)            # a ∘ b
            mul[i][j] = idx[c]
    return elements, mul, id_id


# ----------------------------
# Dataset
# ----------------------------

class RandomSeqFinalDataset(Dataset):
    """
    Random sequences of group elements ids in [0..59].
    Label: final prefix product y_T = x_T ∘ ... ∘ x_1, computed with mul.
    """
    def __init__(self, mul: List[List[int]], id_id: int, length: int, num_samples: int, seed: int = 0):
        super().__init__()
        self.mul = mul
        self.id_id = int(id_id)
        self.length = int(length)
        self.num_samples = int(num_samples)

        rng = random.Random(seed)
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            seq = [rng.randrange(60) for _ in range(length)]
            s = self.id_id
            for g in seq:
                s = self.mul[g][s]  # left multiply: s <- g ∘ s
            self.data.append(seq)
            self.labels.append(s)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = self.labels[idx]
        return {"input_ids": x, "label_final": y}


@torch.no_grad()
def eval_final_acc(model, loader, device, model_name: str,
                   no_scan: bool = False, shuffle_M: bool = False, reset_each_step: bool = False) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["label_final"].to(device)

        if model_name in {"plugin_p0", "select", "exact", "route1"}:
            logits, _ = model(x, labels=None, no_scan=no_scan, shuffle_M=shuffle_M, reset_each_step=reset_each_step)
        else:
            logits, _ = model(x, labels=None)

        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
