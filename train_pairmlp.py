import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from groups import generate_a5
from dataset import build_train_dataset, pad_collate
from model_pairmlp import PairMLPTagger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_len2_pair(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)   # [B,2]
        labels = batch["labels"].to(device)         # [B,2]
        logits, _ = model(input_ids, labels=None)   # [B,60]
        pred = logits.argmax(dim=-1)                # [B]
        y = labels[:, 1]                            # [B]
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    _, mul, id_id = generate_a5()

    # IMPORTANT: train_len=2 and train_random_samples=0 -> should be 3600 all pairs
    train_ds = build_train_dataset(mul, id_id, train_len=2, train_random_samples=0, seed=args.seed)
    print("train_ds size:", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    model = PairMLPTagger(num_tokens=60, hidden=128, mlp_hidden=256, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    model.train()
    for step in range(args.steps + 1):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # only length=2 is expected
            logits, loss = model(input_ids, labels=labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                acc = eval_len2_pair(model, eval_loader, device)
                print(f"step {step} | loss {loss.item():.4f} | acc(pos1) {acc:.4f}")
            step += 1
            if step > args.steps:
                break

    acc = eval_len2_pair(model, eval_loader, device)
    print("final acc(pos1):", acc)


if __name__ == "__main__":
    main()
