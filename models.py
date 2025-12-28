from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


def _compute_prefix_post_states(
    input_ids: torch.Tensor,
    mul: torch.Tensor,
    id_id: int,
    *,
    reset_state: bool = False,
    shuffle_state: bool = False,
) -> torch.Tensor:
    """
    Compute POST-states after consuming each token:
      s_0 = id
      s_{t+1} = x_t ∘ s_t
      post[:, t] = s_{t+1}

    Returns: post_state_ids [B, T]
    """
    B, T = input_ids.shape
    device = input_ids.device

    s = torch.full((B,), int(id_id), device=device, dtype=torch.long)
    post = torch.empty((B, T), device=device, dtype=torch.long)

    for t in range(T):
        if reset_state:
            s.fill_(int(id_id))
        s = mul[input_ids[:, t], s]
        post[:, t] = s

    if shuffle_state and T > 1:
        perm = torch.randperm(T, device=device)
        post = post[:, perm]
    return post


class GPT2FrozenStatePlugin(nn.Module):
    """
    Frozen GPT-2 + teacher-state channel (trainable) for the A5 recurrence task.

    inject_mode:
      - clean: inject PRE-states s_t at position t (anti-leak, last gets s_{T-1})
      - final: only last position gets s_T (oracle)
      - prev : only last position gets s_{T-1}
      - none : inject nothing

    inject_style:
      - input_add: x = tok_emb + masked(state_proj(state_id))
      - fusion    : gated residual hook at inject_layer using the same projected state vector
      - both      : apply both input_add and fusion

    stride_mode (clean only):
      - hold  : hold the injected PRE-state for K steps (piecewise-constant state channel)
      - sparse: only inject at every K-th position (mask=1), optionally randomizing phase during TRAIN

    mid_once (clean only):
      - override stride masking: inject at exactly ONE non-terminal position + always inject at last.
        This is the "minimal bootstrapping" ablation.
    """

    def __init__(
        self,
        mul_table,
        id_id: int,
        *,
        num_tokens: int = 60,
        gpt2_name: str = "openai-community/gpt2",
        inject_layer: int = 8,
        d_state: int = 128,
        local_files_only: bool = False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers not installed. pip install transformers")

        mul = torch.tensor(mul_table, dtype=torch.long) if not torch.is_tensor(mul_table) else mul_table.long()
        self.register_buffer("mul", mul)
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

        self.gpt2 = AutoModel.from_pretrained(
            gpt2_name,
            trust_remote_code=False,
            local_files_only=bool(local_files_only),
        )
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.n_embd = int(self.gpt2.config.n_embd)

        # Task embeddings (trainable, since vocab is synthetic 60-way)
        self.tok_emb = nn.Embedding(self.num_tokens, self.n_embd)

        # Teacher-state channel (trainable)
        self.state_emb = nn.Embedding(self.num_tokens, int(d_state))
        self.state_proj = nn.Linear(int(d_state), self.n_embd)

        # Fusion parameters (trainable)
        self.W_h = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_s = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_d = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.head = nn.Linear(self.n_embd, self.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

        # Hook cache
        self._cached_s: Optional[torch.Tensor] = None  # [B,T,H]
        self._cached_gate_zero: bool = False

        # Find blocks and register hook
        blocks = None
        if hasattr(self.gpt2, "transformer") and hasattr(self.gpt2.transformer, "h"):
            blocks = self.gpt2.transformer.h
        elif hasattr(self.gpt2, "h"):
            blocks = self.gpt2.h
        elif hasattr(self.gpt2, "model") and hasattr(self.gpt2.model, "h"):
            blocks = self.gpt2.model.h
        if blocks is None:
            raise ValueError(f"Cannot locate GPT-2 blocks. Got type={type(self.gpt2)}")

        n_layer = len(blocks)
        self.inject_layer = int(max(0, min(int(inject_layer), n_layer - 1)))
        blocks[self.inject_layer].register_forward_hook(self._fusion_hook)

        print(f"[GPT2FrozenStatePlugin] backbone type: {type(self.gpt2)}")
        print(f"[GPT2FrozenStatePlugin] inject_layer: {self.inject_layer} / n_layer: {n_layer}")

    def _fusion_hook(self, module, inputs, output):
        if self._cached_s is None or self._cached_gate_zero:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        s = self._cached_s

        gate = torch.sigmoid(self.W_h(hidden) + self.W_s(s))
        delta = self.W_d(s)
        hidden2 = hidden + gate * delta

        if isinstance(output, tuple):
            return (hidden2,) + output[1:]
        return hidden2

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        # Train-time ablations (kept, but grouped)
        shuffle_state: bool = False,
        reset_state: bool = False,
        gate_zero: bool = False,
        # State injection
        inject_mode: str = "clean",
        inject_style: str = "input_add",
        # Bandwidth / phase
        state_stride: int = 1,
        stride_mode: str = "hold",
        stride_offset: int = 0,
        random_phase_shift: bool = False,
        phase_shift_mode: str = "batch",
        # Minimal bootstrapping
        mid_once: bool = False,
        mid_pos: int = -1,
        mid_pos_mode: str = "batch",
    ):
        if inject_mode not in {"clean", "final", "prev", "none"}:
            raise ValueError(f"Unknown inject_mode: {inject_mode}")
        if inject_style not in {"input_add", "fusion", "both"}:
            raise ValueError(f"Unknown inject_style: {inject_style}")

        B, T = input_ids.shape
        device = input_ids.device
        attn_mask = torch.ones((B, T), device=device, dtype=torch.long)

        # ---- teacher states ----
        post_state_ids = _compute_prefix_post_states(
            input_ids,
            self.mul,
            self.id_id,
            reset_state=reset_state,
            shuffle_state=shuffle_state,
        )  # [B,T] where post[t] = s_{t+1}

        # PRE states: pre[0]=id, pre[t]=post[t-1] = s_t
        if T > 0:
            pre_state_ids = torch.full((B, T), int(self.id_id), device=device, dtype=torch.long)
            if T > 1:
                pre_state_ids[:, 1:] = post_state_ids[:, :-1]
        else:
            pre_state_ids = post_state_ids

        # ---- choose which state ids to present ----
        if inject_mode == "clean":
            state_ids = pre_state_ids
        else:
            state_ids = torch.full((B, T), int(self.id_id), device=device, dtype=torch.long)
            if T > 0:
                if inject_mode == "final":
                    state_ids[:, -1] = post_state_ids[:, -1]  # s_T
                elif inject_mode == "prev":
                    state_ids[:, -1] = post_state_ids[:, -2] if T >= 2 else int(self.id_id)
                elif inject_mode == "none":
                    pass

        # ---- build injection mask (true no-op when 0) ----
        mask = torch.zeros((B, T, 1), device=device, dtype=torch.float32)
        if inject_mode == "clean":
            mask[:] = 1.0
        elif inject_mode in {"final", "prev"} and T > 0:
            mask[:, -1, 0] = 1.0

        # ---- stride/mid_once (clean only) ----
        K = max(int(state_stride or 1), 1)
        if inject_mode == "clean" and T > 0:
            if mid_once:
                if mid_pos_mode not in {"batch", "sample"}:
                    raise ValueError(f"Unknown mid_pos_mode: {mid_pos_mode}")
                keep = torch.zeros((B, T, 1), device=device, dtype=torch.float32)

                if T >= 2:
                    if mid_pos is not None and int(mid_pos) >= 0:
                        t_mid = int(mid_pos)
                        t_mid = min(t_mid, T - 2)
                        keep[:, t_mid, 0] = 1.0
                    else:
                        if mid_pos_mode == "batch":
                            t_mid = int(torch.randint(low=0, high=T - 1, size=(1,), device=device).item())
                            keep[:, t_mid, 0] = 1.0
                        else:
                            t_mid = torch.randint(low=0, high=T - 1, size=(B,), device=device)
                            keep[torch.arange(B, device=device), t_mid, 0] = 1.0

                keep[:, T - 1, 0] = 1.0  # always show last
                mask = mask * keep
            else:
                if stride_mode not in {"hold", "sparse"}:
                    raise ValueError(f"Unknown stride_mode: {stride_mode}")

                if K > 1 and stride_mode == "hold":
                    held = state_ids.clone()
                    for t0 in range(0, T, K):
                        t1 = min(t0 + K, T)
                        held[:, t0:t1] = state_ids[:, t0:t0 + 1].expand(B, t1 - t0)
                    state_ids = held

                if K > 1 and stride_mode == "sparse":
                    t = torch.arange(T, device=device)
                    offset = int(stride_offset or 0)

                    if random_phase_shift:
                        if phase_shift_mode == "batch":
                            shift = int(torch.randint(low=0, high=K, size=(1,), device=device).item())
                            eff_offset = offset + shift
                            keep = ((t - eff_offset) % K == 0).to(torch.float32)
                            keep[T - 1] = 1.0  # force last visible
                            keep = keep.view(1, T, 1).expand(B, T, 1)
                        elif phase_shift_mode == "sample":
                            shift = torch.randint(low=0, high=K, size=(B, 1, 1), device=device)
                            tt = t.view(1, T, 1).expand(B, T, 1)
                            keep = (((tt - (offset + shift)) % K) == 0).to(torch.float32)
                            keep[:, T - 1, :] = 1.0
                        else:
                            raise ValueError(f"Unknown phase_shift_mode: {phase_shift_mode}")
                    else:
                        keep = ((t - offset) % K == 0).to(torch.float32).view(1, T, 1).expand(B, T, 1)

                    mask = mask * keep

        # ---- project state vectors & apply mask ----
        s = self.state_proj(self.state_emb(state_ids))  # [B,T,H]
        s = s * mask.to(dtype=s.dtype)

        # ---- backbone ----
        x = self.tok_emb(input_ids)
        if inject_style in {"input_add", "both"}:
            x = x + s

        if inject_style in {"fusion", "both"} and mask.sum().item() > 0:
            self._cached_s = s
            self._cached_gate_zero = bool(gate_zero)
        else:
            self._cached_s = None
            self._cached_gate_zero = False

        out = self.gpt2(inputs_embeds=x, attention_mask=attn_mask, use_cache=False, return_dict=True)

        # clear hook cache
        self._cached_s = None
        self._cached_gate_zero = False

        h_last = out.last_hidden_state[:, -1]
        logits = self.head(h_last)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss
