import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from a5_core import generate_a5, RandomSeqFinalDataset, eval_final_acc
from models import BaselineAdapter, GRUBaseline, A5ExactScan, Route1SoftScan


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--model", type=str, default="route1",
                   choices=["adapter", "gru", "exact", "route1"])
    p.add_argument("--d_model", type=int, default=128)

    # adapter
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])

    # gru
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--gru_dropout", type=float, default=0.0)

    # route1
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--aux_weight", type=float, default=5.0)
    p.add_argument("--anneal_aux", action="store_true",
                   help="anneal aux_weight during training via piecewise schedule")

    # optimization
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # data
    p.add_argument("--train_samples", type=int, default=200000)
    p.add_argument("--test_samples_per_len", type=int, default=10000)
    p.add_argument("--schedule", type=str, default="64",
                   help='e.g. "64" or "2,4,8,16,32,64"')
    p.add_argument("--steps_per_stage", type=int, default=5000)
    p.add_argument("--eval_lens", type=str, default="64,128,256,512")

    # logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="outputs")

    # ablations (mechanism evidence)
    p.add_argument("--no_scan", action="store_true")
    p.add_argument("--shuffle_M", action="store_true")
    p.add_argument("--reset_each_step", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    _, mul, id_id = generate_a5()
    device = torch.device(args.device)

    # model
    if args.model == "adapter":
        model = BaselineAdapter(num_tokens=60, d_model=max(args.d_model, 64),
                                mlp_layers=args.mlp_layers, pool=args.pool).to(device)
    elif args.model == "gru":
        model = GRUBaseline(num_tokens=60, d_model=max(args.d_model, 64),
                            num_layers=args.gru_layers, dropout=args.gru_dropout).to(device)
    elif args.model == "exact":
        model = A5ExactScan(mul_table=mul, id_id=id_id, num_tokens=60).to(device)
    elif args.model == "route1":
        model = Route1SoftScan(mul_table=mul, id_id=id_id, num_tokens=60,
                               temp=args.temp, aux_weight=args.aux_weight).to(device)
    else:
        raise ValueError(args.model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    eval_lens = [int(x) for x in args.eval_lens.split(",") if x.strip()]
    eval_loaders = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(mul, id_id, length=L,
                                  num_samples=args.test_samples_per_len, seed=args.seed + 1000 + L)
        eval_loaders[L] = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    log_path = os.path.join(args.out_dir, "log_final.jsonl")
    step = 0

    for stage_len in schedule:
        train_ds = RandomSeqFinalDataset(mul, id_id, length=stage_len,
                                         num_samples=args.train_samples, seed=args.seed + stage_len)
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

            x = batch["input_ids"].to(device)
            y = batch["label_final"].to(device)

            # optional aux anneal for route1
            if args.model == "route1" and args.anneal_aux:
                if step < 500:
                    model._aux_weight_override = args.aux_weight
                elif step < 1000:
                    model._aux_weight_override = 1.0
                elif step < 1500:
                    model._aux_weight_override = 0.2
                else:
                    model._aux_weight_override = 0.0

            if args.model in {"exact", "route1"}:
                logits, loss = model(x, labels=y, no_scan=args.no_scan,
                                     shuffle_M=args.shuffle_M, reset_each_step=args.reset_each_step)
            else:
                logits, loss = model(x, labels=y)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

            if step % args.log_every == 0:
                print(f"step {step} | stage_len {stage_len} | loss {loss.item():.6f}")

            if step % args.eval_every == 0 and step > 0:
                for L, loader in eval_loaders.items():
                    acc = eval_final_acc(model, loader, device, args.model,
                                         no_scan=args.no_scan, shuffle_M=args.shuffle_M, reset_each_step=args.reset_each_step)
                    print(f"[eval] step {step} | model {args.model} | len {L} | final_acc {acc:.4f}")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "step": step,
                            "stage_len": stage_len,
                            "model": args.model,
                            "len": L,
                            "final_acc": acc,
                            "ablation": {
                                "no_scan": args.no_scan,
                                "shuffle_M": args.shuffle_M,
                                "reset_each_step": args.reset_each_step,
                            },
                            "hparams": {
                                "temp": args.temp if args.model == "route1" else None,
                                "aux_weight": args.aux_weight if args.model == "route1" else None,
                                "anneal_aux": args.anneal_aux if args.model == "route1" else None,
                            }
                        }) + "\n")

            step += 1

        print(f"[stage_end] stage_len {stage_len}")
        for L, loader in eval_loaders.items():
            acc = eval_final_acc(model, loader, device, args.model,
                                 no_scan=args.no_scan, shuffle_M=args.shuffle_M, reset_each_step=args.reset_each_step)
            print(f"[eval] step {step} | model {args.model} | len {L} | final_acc {acc:.4f}")

    print(f"Logs written to: {log_path}")


if __name__ == "__main__":
    main()
