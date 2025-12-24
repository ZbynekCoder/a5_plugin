import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from groups import generate_a5
from dataset import RandomSeqFinalDataset
from model_plugin_only_v2 import (
    BaselineAdapter,
    GRUBaseline,
    StateScanPluginP0,
    SelectiveScanPlugin,
    A5ExactScanPlugin,
    A5RouteToGeneratorsPlugin,
    A5RouteToElementPlugin,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_final(model, loader, device, model_name, no_scan, shuffle_M, reset_each_step):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label_final"].to(device)

        if model_name in ["plugin", "select", "exact", "route","route1"]:
            logits, _ = model(
                input_ids,
                labels=None,
                no_scan=no_scan,
                shuffle_M=shuffle_M,
                reset_each_step=reset_each_step,
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

    p.add_argument(
        "--model",
        type=str,
        default="plugin",
        choices=["adapter", "gru", "plugin", "select", "exact", "route","route1"],
    )
    p.add_argument("--d_model", type=int, default=16)

    # adapter
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])

    # gru
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--gru_dropout", type=float, default=0.0)

    # plugin/select readout
    p.add_argument("--readout_hidden", type=int, default=64)

    # selective (Barrington-inspired)
    p.add_argument("--k_generators", type=int, default=16)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--gumbel_hard", action="store_true")

    # route model (token -> generator program)
    p.add_argument("--prog_len", type=int, default=8)
    p.add_argument("--aux_weight", type=float, default=1.0)
    p.add_argument("--use_teacher_words", action="store_true")
    p.add_argument("--g0", type=int, default=-1, help="optional generator id; -1 means auto-search")
    p.add_argument("--g1", type=int, default=-1, help="optional generator id; -1 means auto-search")

    # optimization / data
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_samples", type=int, default=100000)
    p.add_argument("--test_samples_per_len", type=int, default=5000)
    p.add_argument("--eval_lens", type=str, default="64,128,256,512")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="outputs")

    # curriculum schedule: e.g. "64" or "2,4,8,16,32,64"
    p.add_argument("--schedule", type=str, default="64")
    p.add_argument("--steps_per_stage", type=int, default=5000)

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

    eval_lens = [int(x) for x in args.eval_lens.split(",") if x.strip()]
    eval_loaders = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(
            mul, id_id, length=L, num_samples=args.test_samples_per_len, seed=args.seed + L
        )
        eval_loaders[L] = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)

    readout_hidden = None if args.readout_hidden <= 0 else args.readout_hidden

    if args.model == "adapter":
        model = BaselineAdapter(
            num_tokens=60,
            d_model=max(args.d_model, 32),
            mlp_layers=args.mlp_layers,
            pool=args.pool,
        ).to(device)

    elif args.model == "gru":
        model = GRUBaseline(
            num_tokens=60,
            d_model=max(args.d_model, 64),
            num_layers=args.gru_layers,
            dropout=args.gru_dropout,
        ).to(device)

    elif args.model == "select":
        model = SelectiveScanPlugin(
            num_tokens=60,
            d_model=args.d_model,
            k_generators=args.k_generators,
            temp=args.temp,
            gumbel_hard=args.gumbel_hard,
            readout_hidden=readout_hidden,
        ).to(device)

    elif args.model == "exact":
        model = A5ExactScanPlugin(mul_table=mul, id_id=id_id, num_tokens=60).to(device)

    elif args.model == "route":
        g0 = None if args.g0 < 0 else args.g0
        g1 = None if args.g1 < 0 else args.g1
        model = A5RouteToGeneratorsPlugin(
            mul_table=mul,
            id_id=id_id,
            num_tokens=60,
            prog_len=args.prog_len,
            temp=args.temp,
            gumbel_hard=True,  # 强制离散选择，更像“路由”
            aux_weight=args.aux_weight,
            seed=args.seed,
            g0=g0,
            g1=g1,
            use_teacher_words=args.use_teacher_words,
        ).to(device)
        print(f"[route] using generators g0={model.g0}, g1={model.g1}, reachable_with_L={model.reachable}/60, L={model.L}")

    elif args.model == "route1":
        model = A5RouteToElementPlugin(mul_table=mul, id_id=id_id, num_tokens=60, temp=args.temp).to(device)

    else:
        model = StateScanPluginP0(
            num_tokens=60,
            d_model=args.d_model,
            readout_hidden=readout_hidden,
        ).to(device)

    # optimizer：exact 模型没有可训练参数，跳过
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if len(trainable_params) > 0:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

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

            if args.model in ["plugin", "select", "exact", "route","route1"]:
                if args.model == "route1":
                    # piecewise schedule for aux weight (bootstrap then anneal)
                    # You can change these boundaries freely.
                    if step < 500:
                        model._aux_weight_override = 5.0
                    elif step < 1000:
                        model._aux_weight_override = 1.0
                    elif step < 1500:
                        model._aux_weight_override = 0.2
                    else:
                        model._aux_weight_override = 0.0

                logits, loss = model(
                    input_ids,
                    labels=labels,
                    no_scan=args.no_scan,
                    shuffle_M=args.shuffle_M,
                    reset_each_step=args.reset_each_step,
                )
            else:
                logits, loss = model(input_ids, labels=labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
                        args.reset_each_step,
                    )
                    print(f"[eval] step {step} | model {args.model} | len {L} | final_acc {final_acc:.4f}")
                    record = {
                        "step": step,
                        "model": args.model,
                        "len": L,
                        "final_acc": final_acc,
                        "ablation_flags": {
                            "no_scan": args.no_scan,
                            "shuffle_M": args.shuffle_M,
                            "reset_each_step": args.reset_each_step,
                        },
                        "stage_len": stage_len,
                        "hparams": {
                            "prog_len": args.prog_len if args.model == "route" else None,
                            "aux_weight": args.aux_weight if args.model == "route" else None,
                            "temp": args.temp if args.model in ["select", "route"] else None,
                        },
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
                args.reset_each_step,
            )
            print(f"[eval] step {step} | model {args.model} | len {L} | final_acc {final_acc:.4f}")
            record = {
                "step": step,
                "model": args.model,
                "len": L,
                "final_acc": final_acc,
                "ablation_flags": {
                    "no_scan": args.no_scan,
                    "shuffle_M": args.shuffle_M,
                    "reset_each_step": args.reset_each_step,
                },
                "stage_len": stage_len,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        model.train()


if __name__ == "__main__":
    main()
