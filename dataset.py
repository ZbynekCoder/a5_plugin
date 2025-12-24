import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


def make_prefix_labels(seq, mul, id_id):
    prefix = id_id
    labels = []
    for g in seq:
        prefix = mul[g][prefix]  # left multiply: g ∘ prefix
        labels.append(prefix)
    return labels


class FixedPairsDataset(Dataset):
    def __init__(self, mul, id_id):
        pairs = []
        labels = []
        for a in range(60):
            for b in range(60):
                seq = [a, b]
                pairs.append(seq)
                labels.append(make_prefix_labels(seq, mul, id_id))
        self.inputs = torch.tensor(pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}


class RandomSeqDataset(Dataset):
    def __init__(self, mul, id_id, length, num_samples, seed):
        rng = np.random.RandomState(seed)
        seqs = rng.randint(0, 60, size=(num_samples, length), dtype=np.int64)

        labels = []
        for i in range(num_samples):
            seq = seqs[i].tolist()
            labels.append(make_prefix_labels(seq, mul, id_id))

        self.inputs = torch.tensor(seqs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}


class RandomSeqFinalDataset(Dataset):
    def __init__(self, mul, id_id, length, num_samples, seed):
        rng = np.random.RandomState(seed)
        seqs = rng.randint(0, 60, size=(num_samples, length), dtype=np.int64)

        labels = []
        for i in range(num_samples):
            seq = seqs[i].tolist()
            prefix = id_id
            for g in seq:
                prefix = mul[g][prefix]  # left multiply: g ∘ prefix
            labels.append(prefix)

        self.inputs = torch.tensor(seqs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "label_final": self.labels[idx]}


def build_train_dataset(mul, id_id, train_len, train_random_samples, seed):
    ds_pairs = FixedPairsDataset(mul, id_id)
    ds_random = RandomSeqDataset(
        mul, id_id, length=train_len, num_samples=train_random_samples, seed=seed
    )
    return ConcatDataset([ds_pairs, ds_random])


def pad_collate(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    bsz = len(batch)

    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        if seq_len > 1:
            labels[i, 1:seq_len] = item["labels"][1:seq_len]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
