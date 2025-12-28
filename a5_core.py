from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# ----------------------------
# A5 group as even permutations on 5 letters
# ----------------------------

Perm = Tuple[int, ...]


def _compose(p: Perm, q: Perm) -> Perm:
    """Permutation composition: (p ∘ q)(i) = p[q[i]]"""
    return tuple(p[q[i]] for i in range(len(p)))


def _parity(perm: Perm) -> int:
    """Return 0 if even, 1 if odd."""
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv % 2


def generate_a5() -> Tuple[List[Perm], List[List[int]], int]:
    """
    Generate A5 as even permutations on 5 letters.

    Returns:
      elems: list of 60 permutations (tuples)
      mul: Cayley table ids: mul[g][s] = g ∘ s   (LEFT-multiply)
      id_id: identity element id
    """
    perms = list(itertools.permutations(range(5)))
    elems = [p for p in perms if _parity(p) == 0]
    assert len(elems) == 60

    idx = {p: i for i, p in enumerate(elems)}
    id_perm = tuple(range(5))
    id_id = idx[id_perm]

    mul = [[0] * 60 for _ in range(60)]
    for i, p in enumerate(elems):
        for j, q in enumerate(elems):
            r = _compose(p, q)
            mul[i][j] = idx[r]
    return elems, mul, id_id


# ----------------------------
# Dataset: random sequences, label is final prefix product
# ----------------------------

class RandomSeqFinalDataset(Dataset):
    """
    Random sequences of group element ids in [0..59].
    Label: final prefix product s_T computed by:
      s_0 = id
      s_{t+1} = x_t ∘ s_t   (LEFT-multiply)
    """

    def __init__(self, mul: List[List[int]], id_id: int, length: int, num_samples: int, seed: int = 0):
        super().__init__()
        self.mul = mul
        self.id_id = int(id_id)
        self.length = int(length)
        self.num_samples = int(num_samples)

        rng = random.Random(seed)
        self.data: List[List[int]] = []
        labels: List[int] = []
        for _ in range(num_samples):
            seq = [rng.randrange(60) for _ in range(self.length)]
            s = self.id_id
            for g in seq:
                s = self.mul[g][s]  # s <- g ∘ s
            self.data.append(seq)
            labels.append(s)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = self.labels[idx]
        return {"input_ids": x, "label_final": y}


@torch.no_grad()
def eval_final_acc(
    model,
    loader,
    device,
    *,
    inject_mode: str,
    inject_style: str,
    state_stride: int,
    stride_mode: str,
    stride_offset: int,
    random_phase_shift: bool,
    phase_shift_mode: str,
    mid_once: bool,
    mid_pos: int,
    mid_pos_mode: str,
) -> float:
    """Accuracy of predicting final group state s_T."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["label_final"].to(device)
        logits, _ = model(
            x,
            labels=None,
            inject_mode=inject_mode,
            inject_style=inject_style,
            state_stride=state_stride,
            stride_mode=stride_mode,
            stride_offset=stride_offset,
            random_phase_shift=random_phase_shift,
            phase_shift_mode=phase_shift_mode,
            mid_once=mid_once,
            mid_pos=mid_pos,
            mid_pos_mode=mid_pos_mode,
        )
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
