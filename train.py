import os

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from a5_core import generate_a5, RandomSeqFinalDataset, eval_final_acc
from models import (
    BaselineAdapter,
    GRUBaseline,
    A5ExactScan,
    Route1SoftScan,
    GPT2FrozenBaseline,
    GPT2FrozenStateFusion,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()

    # basic
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    # model selection
    p.add_argument(
        "--model",
        type=str,
        default="route1",
        choices=["adapter", "gru", "exact", "route1", "gpt2", "gpt2_state"],
    )
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
    p.add_argument("--anneal_aux", action="store_true")

    # gpt2 frozen / state plugin
    p.add_argument("--gpt2_name", type=str, default="openai-community/gpt2")
    p.add_argument("--inject_layer", type=int, default=8)
    p.add_argument("--d_state", type=int, default=128)
    p.add_argument("--inject_style", type=str, default="input_add", choices=["input_add", "fusion", "both"])
    p.add_argument(
        "--state_stride",
        type=int,
        default=1,
        help="For gpt2_state (inject_mode=clean): hold the injected PRE-state for K steps (K>=1).",
    )
    p.add_argument("--stride_mode", type=str, default="hold", choices=["hold", "sparse"])
    p.add_argument("--stride_offset", type=int, default=0)
    # ---- sparse stride phase randomization ----
    p.add_argument(
        "--random_phase_shift",
        action="store_true",
        help="(gpt2_state + stride_mode=sparse + inject_mode=clean) randomize the sparse injection phase each batch/sample during TRAIN. Disabled by default.",
    )
    p.add_argument(
        "--phase_shift_mode",
        type=str,
        default="batch",
        choices=["batch", "sample"],
        help="How to randomize phase when --random_phase_shift is enabled. batch=one shift per batch, sample=independent shift per sample.",
    )

    p.add_argument("--local_files_only", action="store_true")

    # ---- TRAIN-TIME ablations ----
    p.add_argument("--shuffle_state", action="store_true")
    p.add_argument("--reset_state", action="store_true")
    p.add_argument("--gate_zero", action="store_true")

    # Train-time injection mode
    p.add_argument(
        "--train_inject_mode",
        type=str,
        default="clean",
        choices=["clean", "final", "prev", "none"],
        help="Train-time injection mode for gpt2_state. clean=PRE-states (anti-leak). final is oracle leak.",
    )

    # ---- EVAL bundle ----
    p.add_argument("--eval_multi", action="store_true")
    p.add_argument("--eval_inject_modes", type=str, default="clean,final,prev,none")

    p.add_argument("--eval_gate_zero", action="store_true")
    p.add_argument("--eval_shuffle_state", action="store_true")
    p.add_argument("--eval_reset_state", action="store_true")

    # optimization
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # data
    p.add_argument("--train_samples", type=int, default=200000)
    p.add_argument("--test_samples_per_len", type=int, default=10000)
    p.add_argument("--schedule", type=str, default="64")
    p.add_argument("--steps_per_stage", type=int, default=5000)
    p.add_argument("--eval_lens", type=str, default="64,128,256,512")

    # logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="outputs")

    # mechanism ablations for exact/route1 (executor-side)
    p.add_argument("--no_scan", action="store_true")
    p.add_argument("--shuffle_M", action="store_true")
    p.add_argument("--reset_each_step", action="store_true")

    return p.parse_args()


def build_model(args, mul, id_id, device):
    if args.model == "adapter":
        return BaselineAdapter(
            num_tokens=60,
            d_model=max(args.d_model, 64),
            mlp_layers=args.mlp_layers,
            pool=args.pool,
        ).to(device)

    if args.model == "gru":
        return GRUBaseline(
            num_tokens=60,
            d_model=max(args.d_model, 64),
            num_layers=args.gru_layers,
            dropout=args.gru_dropout,
        ).to(device)

    if args.model == "exact":
        return A5ExactScan(mul_table=mul, id_id=id_id, num_tokens=60).to(device)

    if args.model == "route1":
        return Route1SoftScan(
            mul_table=mul,
            id_id=id_id,
            num_tokens=60,
            temp=args.temp,
            aux_weight=args.aux_weight,
        ).to(device)

    if args.model == "gpt2":
        return GPT2FrozenBaseline(
            num_tokens=60,
            gpt2_name=args.gpt2_name,
            local_files_only=args.local_files_only,
        ).to(device)

    if args.model == "gpt2_state":
        return GPT2FrozenStateFusion(
            mul_table=mul,
            id_id=id_id,
            num_tokens=60,
            gpt2_name=args.gpt2_name,
            inject_layer=args.inject_layer,
            d_state=args.d_state,
            local_files_only=args.local_files_only,
        ).to(device)

    raise ValueError(args.model)


def train_step(model, args, x, y):
    if args.model in {"exact", "route1"}:
        return model(
            x,
            labels=y,
            no_scan=args.no_scan,
            shuffle_M=args.shuffle_M,
            reset_each_step=args.reset_each_step,
        )

    if args.model == "gpt2_state":
        return model(
            x,
            labels=y,
            shuffle_state=args.shuffle_state,
            reset_state=args.reset_state,
            gate_zero=args.gate_zero,
            state_stride=args.state_stride,
            stride_mode=args.stride_mode,
            stride_offset=args.stride_offset,
            inject_mode=args.train_inject_mode,
            inject_style=args.inject_style,
            random_phase_shift=args.random_phase_shift,
            phase_shift_mode=args.phase_shift_mode,
        )

    return model(x, labels=y)


def _parse_eval_inject_modes(s: str):
    modes = []
    for t in s.split(","):
        m = t.strip()
        if not m:
            continue
        if m not in {"clean", "final", "prev", "none"}:
            raise ValueError(f"Unknown inject mode: {m}")
        modes.append(m)
    if not modes:
        modes = ["clean"]

    seen = set()
    uniq = []
    for m in modes:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


@torch.no_grad()
def run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len):
    eval_modes = ["clean"]
    if args.model == "gpt2_state" and args.eval_multi:
        eval_modes = _parse_eval_inject_modes(args.eval_inject_modes)

    eval_confs = []
    for mode in eval_modes:
        eval_confs.append((mode, dict(inject_mode=mode, shuffle_state=False, reset_state=False, gate_zero=False)))

    if args.model == "gpt2_state" and args.eval_multi:
        if args.eval_gate_zero:
            eval_confs.append(("clean+gate0", dict(inject_mode="clean", shuffle_state=False, reset_state=False, gate_zero=True)))
        if args.eval_shuffle_state:
            eval_confs.append(("clean+shuffle", dict(inject_mode="clean", shuffle_state=True, reset_state=False, gate_zero=False)))
        if args.eval_reset_state:
            eval_confs.append(("clean+reset", dict(inject_mode="clean", shuffle_state=False, reset_state=True, gate_zero=False)))

    for eval_tag, conf in eval_confs:
        for L, loader in eval_loaders.items():
            acc = eval_final_acc(
                model,
                loader,
                device,
                args.model,
                no_scan=args.no_scan,
                shuffle_M=args.shuffle_M,
                reset_each_step=args.reset_each_step,
                shuffle_state=conf["shuffle_state"],
                reset_state=conf["reset_state"],
                gate_zero=conf["gate_zero"],
                state_stride=args.state_stride,
                stride_mode=args.stride_mode,
                stride_offset=args.stride_offset,
                inject_mode=conf["inject_mode"],
                inject_style=args.inject_style,
                random_phase_shift=False,
                phase_shift_mode="batch",
            )

            print(f"[eval] step {step} | model {args.model} | style {args.inject_style} | tag {eval_tag} | len {L} | final_acc {acc:.4f}")

            rec = {
                "type": "eval",
                "step": step,
                "stage_len": stage_len,
                "model": args.model,
                "inject_style": args.inject_style,
                "eval_tag": eval_tag,
                "len": L,
                "final_acc": acc,
                "train_time": {
                    "train_inject_mode": args.train_inject_mode,
                    "shuffle_state": args.shuffle_state,
                    "reset_state": args.reset_state,
                    "gate_zero": args.gate_zero,
                },
                "eval_conf": conf,
                "hparams": {
                    "gpt2_name": args.gpt2_name if args.model in {"gpt2", "gpt2_state"} else None,
                    "inject_layer": args.inject_layer if args.model == "gpt2_state" else None,
                    "d_state": args.d_state if args.model == "gpt2_state" else None,
                    "state_stride": args.state_stride if args.model == "gpt2_state" else None,
                },
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    _, mul, id_id = generate_a5()
    device = torch.device(args.device)

    model = build_model(args, mul, id_id, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    eval_lens = [int(x) for x in args.eval_lens.split(",") if x.strip()]
    eval_loaders = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(mul, id_id, length=L, num_samples=args.test_samples_per_len, seed=args.seed + 100 + L)
        eval_loaders[L] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    assert len(schedule) >= 1

    log_path = os.path.join(args.out_dir, "log_final.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    win_correct = 0
    win_total = 0

    step = 0
    for stage_len in schedule:
        ds_train = RandomSeqFinalDataset(mul, id_id, length=stage_len, num_samples=args.train_samples, seed=args.seed + stage_len)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        print(f"[stage_start] stage_len {stage_len} | steps {args.steps_per_stage}")

        it = iter(train_loader)
        for _ in range(args.steps_per_stage):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            x = batch["input_ids"].to(device)
            y = batch["label_final"].to(device)

            logits, loss = train_step(model, args, x, y)

            pred = logits.argmax(dim=-1)
            win_correct += (pred == y).sum().item()
            win_total += y.numel()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

            if step % args.log_every == 0:
                train_acc = win_correct / max(win_total, 1)
                print(f"step {step} | stage_len {stage_len} | loss {loss.item():.6f} | train_acc(win) {train_acc:.4f}")

                win_correct = 0
                win_total = 0

            if step % args.eval_every == 0 and step > 0:
                run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len)

            step += 1

        print(f"[stage_end] stage_len {stage_len}")
        run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len)

    print(f"Logs written to: {log_path}")


if __name__ == "__main__":
    main()
