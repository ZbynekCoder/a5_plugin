import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from groups import generate_a5
from dataset import RandomSeqFinalDataset
from model_plugin_only import BaselineAdapter, StateScanPlugin


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_final(model, loader, device, model_name, no_scan, shuffle_M, zero_state):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label_final"].to(device)

        if model_name == "plugin":
            logits, _ = model(
                input_ids,
                labels=None,
                no_scan=no_scan,
                shuffle_M=shuffle_M,
                zero_state=zero_state,
            )
        else:
            logits, _ = model(input_ids, labels=None)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(total, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model", type=str, default="plugin", choices=["adapter", "plugin"])
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    p.add_argument("--eps", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_samples", type=int, default=50000)
    p.add_argument("--test_samples_per_len", type=int, default=2000)
    p.add_argument("--eval_lens", type=str, default="64")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="outputs")

    p.add_argument("--schedule", type=str, default="2,4,8,16,32,64")
    p.add_argument("--steps_per_stage", type=int, default=2000)

    p.add_argument("--no_scan", action="store_true")
    p.add_argument("--shuffle_M", action="store_true")
    p.add_argument("--zero_state", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    _, mul, id_id = generate_a5()

    eval_lens = [int(x) for x in args.eval_lens.split(",") if x.strip()]
    eval_loaders = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(
            mul, id_id, length=L, num_samples=args.test_samples_per_len, seed=args.seed + L
        )
        eval_loaders[L] = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    if args.model == "adapter":
        model = BaselineAdapter(
            num_tokens=60,
            d_model=args.d_model,
            mlp_layers=args.mlp_layers,
            pool=args.pool,
        ).to(device)
    else:
        model = StateScanPlugin(
            num_tokens=60,
            d_model=args.d_model,
            eps=args.eps,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    log_path = os.path.join(args.out_dir, "log_final.jsonl")
    step = 0
    model.train()

    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    for stage_len in schedule:
        train_ds = RandomSeqFinalDataset(
            mul, id_id, length=stage_len, num_samples=args.train_samples, seed=args.seed + stage_len
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        labels_preview = train_ds.labels[:10].tolist()
        labels_unique = torch.unique(train_ds.labels).numel()
        print(f"[stage_start] stage_len {stage_len} | labels_head {labels_preview} | unique {labels_unique}")

        it = iter(train_loader)
        for _ in range(args.steps_per_stage):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            input_ids = batch["input_ids"].to(device)
            labels = batch["label_final"].to(device)

            logits, loss = model(input_ids, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                print(f"step {step} | stage_len {stage_len} | loss {loss.item():.6f}")

            if step % args.eval_every == 0 and step > 0:
                model.eval()
                for L, loader in eval_loaders.items():
                    final_acc = evaluate_final(
                        model,
                        loader,
                        device,
                        args.model,
                        args.no_scan,
                        args.shuffle_M,
                        args.zero_state,
                    )
                    print(
                        f"[eval] step {step} | model {args.model} | len {L} | final_acc {final_acc:.4f}"
                    )
                    record = {
                        "step": step,
                        "model": args.model,
                        "len": L,
                        "final_acc": final_acc,
                        "ablation_flags": {
                            "no_scan": args.no_scan,
                            "shuffle_M": args.shuffle_M,
                            "zero_state": args.zero_state,
                        },
                        "stage_len": stage_len,
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                model.train()

            step += 1

        print(f"[stage_end] stage_len {stage_len}")
        model.eval()
        for L, loader in eval_loaders.items():
            final_acc = evaluate_final(
                model,
                loader,
                device,
                args.model,
                args.no_scan,
                args.shuffle_M,
                args.zero_state,
            )
            print(
                f"[eval] step {step} | model {args.model} | len {L} | final_acc {final_acc:.4f}"
            )
            record = {
                "step": step,
                "model": args.model,
                "len": L,
                "final_acc": final_acc,
                "ablation_flags": {
                    "no_scan": args.no_scan,
                    "shuffle_M": args.shuffle_M,
                    "zero_state": args.zero_state,
                },
                "stage_len": stage_len,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        model.train()


if __name__ == "__main__":
    main()
