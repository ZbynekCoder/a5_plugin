from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from a5_core import RandomSeqFinalDataset, eval_final_acc, generate_a5
from models import GPT2FrozenStatePlugin


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class HParams:
    # basics
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "outputs"

    # data
    train_samples: int = 200_000
    test_samples_per_len: int = 10_000
    schedule: str = "64"               # training lengths, comma-separated
    steps_per_stage: int = 1000
    eval_lens: str = "64,128,256,512"

    # optimization
    batch_size: int = 512
    lr: float = 5e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # model
    gpt2_name: str = "openai-community/gpt2"
    local_files_only: bool = False
    inject_layer: int = 8
    d_state: int = 128
    inject_style: str = "input_add"    # input_add|fusion|both

    # train-time injection
    train_inject_mode: str = "clean"   # clean|final|prev|none
    shuffle_state: bool = False
    reset_state: bool = False
    gate_zero: bool = False

    # bandwidth / phase (clean)
    state_stride: int = 1
    stride_mode: str = "hold"          # hold|sparse
    stride_offset: int = 0
    random_phase_shift: bool = False
    phase_shift_mode: str = "batch"    # batch|sample

    # minimal bootstrapping (clean)
    mid_once: bool = False
    mid_pos: int = -1                  # 0..T-2, -1 => random
    mid_pos_mode: str = "batch"        # batch|sample

    # eval
    eval_every: int = 200
    log_every: int = 100
    eval_inject_modes: str = "clean,final"  # comma-separated


def parse_args() -> HParams:
    p = argparse.ArgumentParser("A5 state-channel learnability (frozen GPT-2 + teacher-state plugin)")

    # basics
    p.add_argument("--seed", type=int, default=HParams.seed)
    p.add_argument("--device", type=str, default=HParams.device)
    p.add_argument("--out_dir", type=str, default=HParams.out_dir)

    # data
    p.add_argument("--train_samples", type=int, default=HParams.train_samples)
    p.add_argument("--test_samples_per_len", type=int, default=HParams.test_samples_per_len)
    p.add_argument("--schedule", type=str, default=HParams.schedule)
    p.add_argument("--steps_per_stage", type=int, default=HParams.steps_per_stage)
    p.add_argument("--eval_lens", type=str, default=HParams.eval_lens)

    # optimization
    p.add_argument("--batch_size", type=int, default=HParams.batch_size)
    p.add_argument("--lr", type=float, default=HParams.lr)
    p.add_argument("--weight_decay", type=float, default=HParams.weight_decay)
    p.add_argument("--grad_clip", type=float, default=HParams.grad_clip)

    # model
    p.add_argument("--gpt2_name", type=str, default=HParams.gpt2_name)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--inject_layer", type=int, default=HParams.inject_layer)
    p.add_argument("--d_state", type=int, default=HParams.d_state)
    p.add_argument("--inject_style", type=str, default=HParams.inject_style, choices=["input_add", "fusion", "both"])

    # train-time injection
    p.add_argument("--train_inject_mode", type=str, default=HParams.train_inject_mode, choices=["clean", "final", "prev", "none"])
    p.add_argument("--shuffle_state", action="store_true")
    p.add_argument("--reset_state", action="store_true")
    p.add_argument("--gate_zero", action="store_true")

    # bandwidth / phase
    p.add_argument("--state_stride", type=int, default=HParams.state_stride)
    p.add_argument("--stride_mode", type=str, default=HParams.stride_mode, choices=["hold", "sparse"])
    p.add_argument("--stride_offset", type=int, default=HParams.stride_offset)
    p.add_argument("--random_phase_shift", action="store_true")
    p.add_argument("--phase_shift_mode", type=str, default=HParams.phase_shift_mode, choices=["batch", "sample"])

    # minimal bootstrapping
    p.add_argument("--mid_once", action="store_true")
    p.add_argument("--mid_pos", type=int, default=HParams.mid_pos)
    p.add_argument("--mid_pos_mode", type=str, default=HParams.mid_pos_mode, choices=["batch", "sample"])

    # logging / eval
    p.add_argument("--eval_every", type=int, default=HParams.eval_every)
    p.add_argument("--log_every", type=int, default=HParams.log_every)
    p.add_argument("--eval_inject_modes", type=str, default=HParams.eval_inject_modes)

    args = p.parse_args()
    return HParams(**vars(args))


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_eval_modes(s: str) -> List[str]:
    out: List[str] = []
    for t in s.split(","):
        m = t.strip()
        if not m:
            continue
        if m not in {"clean", "final", "prev", "none"}:
            raise ValueError(f"Unknown inject mode: {m}")
        if m not in out:
            out.append(m)
    return out or ["clean"]


@torch.no_grad()
def run_eval(model, hp: HParams, eval_loaders: Dict[int, DataLoader], device: torch.device, step: int, stage_len: int, log_path: str) -> None:
    eval_modes = parse_eval_modes(hp.eval_inject_modes)

    for inject_mode in eval_modes:
        for L, loader in eval_loaders.items():
            acc = eval_final_acc(
                model,
                loader,
                device,
                inject_mode=inject_mode,
                inject_style=hp.inject_style,
                state_stride=hp.state_stride,
                stride_mode=hp.stride_mode,
                stride_offset=hp.stride_offset,
                random_phase_shift=False,      # eval: deterministic
                phase_shift_mode="batch",
                mid_once=hp.mid_once,
                mid_pos=hp.mid_pos,
                mid_pos_mode=hp.mid_pos_mode,
            )

            print(f"[eval] step {step} | stage_len {stage_len} | inject_mode {inject_mode} | len {L} | acc {acc:.4f}")

            rec = {
                "type": "eval",
                "step": step,
                "stage_len": stage_len,
                "inject_mode": inject_mode,
                "len": L,
                "acc": acc,
                "hparams": asdict(hp),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    hp = parse_args()
    os.makedirs(hp.out_dir, exist_ok=True)
    set_seed(hp.seed)

    _, mul, id_id = generate_a5()
    device = torch.device(hp.device)

    model = GPT2FrozenStatePlugin(
        mul_table=mul,
        id_id=id_id,
        gpt2_name=hp.gpt2_name,
        inject_layer=hp.inject_layer,
        d_state=hp.d_state,
        local_files_only=hp.local_files_only,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

    eval_lens = parse_csv_ints(hp.eval_lens)
    eval_loaders: Dict[int, DataLoader] = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(mul, id_id, length=L, num_samples=hp.test_samples_per_len, seed=hp.seed + 100 + L)
        eval_loaders[L] = DataLoader(ds, batch_size=hp.batch_size, shuffle=False, drop_last=False)

    schedule = parse_csv_ints(hp.schedule)
    assert len(schedule) >= 1

    log_path = os.path.join(hp.out_dir, "log_final.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    win_correct = 0
    win_total = 0
    step = 0

    for stage_len in schedule:
        ds_train = RandomSeqFinalDataset(mul, id_id, length=stage_len, num_samples=hp.train_samples, seed=hp.seed + stage_len)
        train_loader = DataLoader(ds_train, batch_size=hp.batch_size, shuffle=True, drop_last=True)

        print(f"[stage_start] stage_len {stage_len} | steps {hp.steps_per_stage}")

        it = iter(train_loader)
        for _ in range(hp.steps_per_stage):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            x = batch["input_ids"].to(device)
            y = batch["label_final"].to(device)

            logits, loss = model(
                x,
                labels=y,
                inject_mode=hp.train_inject_mode,
                inject_style=hp.inject_style,
                shuffle_state=hp.shuffle_state,
                reset_state=hp.reset_state,
                gate_zero=hp.gate_zero,
                state_stride=hp.state_stride,
                stride_mode=hp.stride_mode,
                stride_offset=hp.stride_offset,
                random_phase_shift=hp.random_phase_shift,
                phase_shift_mode=hp.phase_shift_mode,
                mid_once=hp.mid_once,
                mid_pos=hp.mid_pos,
                mid_pos_mode=hp.mid_pos_mode,
            )

            pred = logits.argmax(dim=-1)
            win_correct += (pred == y).sum().item()
            win_total += y.numel()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, hp.grad_clip)
            optimizer.step()

            if step % hp.log_every == 0:
                train_acc = win_correct / max(win_total, 1)
                print(f"step {step} | stage_len {stage_len} | loss {loss.item():.6f} | train_acc(win) {train_acc:.4f}")
                win_correct = 0
                win_total = 0

            if step % hp.eval_every == 0 and step > 0:
                run_eval(model, hp, eval_loaders, device, step, stage_len, log_path)

            step += 1

        print(f"[stage_end] stage_len {stage_len}")
        run_eval(model, hp, eval_loaders, device, step, stage_len, log_path)

    print(f"Logs written to: {log_path}")


if __name__ == "__main__":
    main()
