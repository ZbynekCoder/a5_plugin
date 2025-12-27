from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


# ----------------------------
# Baselines
# ----------------------------

class BaselineAdapter(nn.Module):
    def __init__(self, num_tokens: int = 60, d_model: int = 64, mlp_layers: int = 2, pool: str = "mean"):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        layers = []
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.pool = pool
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        x = self.mlp(x)
        h = x.mean(dim=1) if self.pool == "mean" else x[:, -1]
        logits = self.head(h)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


class GRUBaseline(nn.Module):
    def __init__(self, num_tokens: int = 60, d_model: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        _, h = self.gru(x)
        logits = self.head(h[-1])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


# ----------------------------
# Exact executor (Cayley scan)
# ----------------------------

class A5ExactScan(nn.Module):
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        self.register_buffer("mul", mul_table.long())
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        mul = self.mul
        if shuffle_M:
            perm = torch.randperm(self.num_tokens, device=device)
            mul = mul[perm][:, perm]

        s = torch.full((B,), self.id_id, device=device, dtype=torch.long)
        if not no_scan:
            for t in range(T):
                if reset_each_step:
                    s.fill_(self.id_id)
                g = input_ids[:, t]
                s = mul[g, s]

        logits_final = torch.full((B, self.num_tokens), -50.0, device=device)
        logits_final.scatter_(1, s.view(-1, 1), 0.0)

        loss = self.loss_fn(logits_final, labels) if labels is not None else None
        return logits_final, loss


# ----------------------------
# Route1: router + exact executor (soft scan)
# ----------------------------

class Route1SoftScan(nn.Module):
    def __init__(
            self,
            mul_table,
            id_id: int,
            num_tokens: int = 60,
            d_model: int = 128,
            temp: float = 1.0,
            aux_weight: float = 5.0,
    ):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        self.register_buffer("mul", mul_table.long())
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

        self.embed = nn.Embedding(num_tokens, d_model)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_tokens),
        )

        self.temp = float(temp)
        self.aux_weight = float(aux_weight)
        self._aux_weight_override = None
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        mul = self.mul
        if shuffle_M:
            perm = torch.randperm(self.num_tokens, device=device)
            mul = mul[perm][:, perm]

        h = self.embed(input_ids)
        logits_g = self.router(h)
        p_g = F.softmax(logits_g / self.temp, dim=-1)

        s_dist = torch.zeros((B, self.num_tokens), device=device)
        s_dist[:, self.id_id] = 1.0

        if not no_scan:
            for t in range(T):
                if reset_each_step:
                    s_dist.zero_()
                    s_dist[:, self.id_id] = 1.0
                pg = p_g[:, t]
                next_s = torch.zeros_like(s_dist)
                for g in range(self.num_tokens):
                    dest = mul[g]
                    next_s.scatter_add_(
                        1,
                        dest.view(1, -1).expand(B, -1),
                        (pg[:, g].view(B, 1) * s_dist),
                    )
                s_dist = next_s

        logits_final = (s_dist.clamp_min(1e-9)).log()

        loss = None
        if labels is not None:
            loss_final = self.loss_fn(logits_final, labels)
            loss_route = F.cross_entropy(logits_g.reshape(-1, self.num_tokens), input_ids.reshape(-1))
            w = self.aux_weight if self._aux_weight_override is None else float(self._aux_weight_override)
            loss = loss_final + w * loss_route

        return logits_final, loss


# ============================
# Frozen GPT-2 baselines + Teacher-state injection
# ============================

def _compute_prefix_states(
        input_ids: torch.Tensor,
        mul: torch.Tensor,
        id_id: int,
        shuffle_state: bool = False,
        reset_state: bool = False,
) -> torch.Tensor:
    """
    POST-states after consuming token t:
      s_0 = id
      s_{t+1} = x_t ∘ s_t
      states[:, t] = s_{t+1}
    """
    B, T = input_ids.shape
    device = input_ids.device
    s = torch.full((B,), int(id_id), device=device, dtype=torch.long)
    states = torch.empty((B, T), device=device, dtype=torch.long)

    for t in range(T):
        if reset_state:
            s.fill_(int(id_id))
        s = mul[input_ids[:, t], s]
        states[:, t] = s

    if shuffle_state and T > 1:
        perm = torch.randperm(T, device=device)
        states = states[:, perm]
    return states


class GPT2FrozenBaseline(nn.Module):
    def __init__(
            self,
            num_tokens: int = 60,
            gpt2_name: str = "openai-community/gpt2",
            local_files_only: bool = False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers not installed. pip install transformers")

        self.gpt2 = AutoModel.from_pretrained(
            gpt2_name,
            trust_remote_code=False,
            local_files_only=bool(local_files_only),
        )
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.n_embd = int(self.gpt2.config.n_embd)
        self.num_tokens = int(num_tokens)

        self.tok_emb = nn.Embedding(self.num_tokens, self.n_embd)
        self.head = nn.Linear(self.n_embd, self.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        attn_mask = torch.ones((B, T), device=input_ids.device, dtype=torch.long)

        x = self.tok_emb(input_ids)
        out = self.gpt2(inputs_embeds=x, attention_mask=attn_mask, use_cache=False, return_dict=True)
        h_last = out.last_hidden_state[:, -1]
        logits = self.head(h_last)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


class GPT2FrozenStateFusion(nn.Module):
    """
    Frozen GPT-2 + teacher state injection.

    inject_mode semantics (IMPORTANT):
      - clean: inject PRE-states (pre_state[t] = s_{t-1}), i.e. last pos gets s_{T-1} (anti-leak)
      - final: only last pos gets s_T (oracle)
      - prev : only last pos gets s_{T-1}
      - none : inject nothing

    inject_style:
      - input_add: x = tok_emb + (masked state)
      - fusion    : use gated residual fusion hook at inject_layer
      - both      : do both (debug / sanity)

    mid_once (clean-only):
      - override mask to inject at exactly ONE non-terminal position + always inject last position.
    """

    def __init__(
            self,
            mul_table,
            id_id: int,
            num_tokens: int = 60,
            gpt2_name: str = "openai-community/gpt2",
            inject_layer: int = 8,
            d_state: int = 128,
            local_files_only: bool = False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers not installed. pip install transformers")

        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.long()
        self.register_buffer("mul", mul_table)
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

        self.tok_emb = nn.Embedding(self.num_tokens, self.n_embd)
        self.state_emb = nn.Embedding(self.num_tokens, int(d_state))
        self.state_proj = nn.Linear(int(d_state), self.n_embd)

        # fusion params
        self.W_h = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_s = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_d = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.head = nn.Linear(self.n_embd, self.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

        self._cached_s = None
        self._cached_gate_zero = False

        # locate transformer blocks
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
        print(f"[GPT2FrozenStateFusion] backbone type: {type(self.gpt2)}")
        print(f"[GPT2FrozenStateFusion] inject_layer: {self.inject_layer} / n_layer: {n_layer}")

    def _fusion_hook(self, module, inputs, output):
        if self._cached_s is None or self._cached_gate_zero:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        s = self._cached_s  # [B,T,H]

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
            shuffle_state: bool = False,
            reset_state: bool = False,
            gate_zero: bool = False,
            state_stride: int = 1,
            stride_mode: str = "hold",
            stride_offset: int = 0,
            inject_mode: str = "clean",
            inject_style: str = "input_add",
            random_phase_shift: bool = False,
            phase_shift_mode: str = "batch",
            # ---- mid-once injection (clean-only) ----
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

        # ---- compute teacher states ----
        post_state_ids = _compute_prefix_states(
            input_ids,
            self.mul,
            self.id_id,
            shuffle_state=shuffle_state,
            reset_state=reset_state,
        )  # [B,T] where post[t]=s_{t+1}

        # PRE states: pre[0]=id, pre[t]=post[t-1] = s_t
        if T > 0:
            pre_state_ids = torch.full((B, T), int(self.id_id), device=device, dtype=torch.long)
            if T > 1:
                pre_state_ids[:, 1:] = post_state_ids[:, :-1]
        else:
            pre_state_ids = post_state_ids

        # ---- choose state ids ----
        if inject_mode == "clean":
            state_ids = pre_state_ids
        else:
            state_ids = torch.full((B, T), int(self.id_id), device=device, dtype=torch.long)
            if T > 0:
                if inject_mode == "final":
                    state_ids[:, -1] = post_state_ids[:, -1]  # s_T (oracle)
                elif inject_mode == "prev":
                    state_ids[:, -1] = post_state_ids[:, -2] if T >= 2 else int(self.id_id)
                elif inject_mode == "none":
                    pass

        # ---- stride handling: hold vs sparse (only meaningful for clean) ----
        K = int(state_stride) if state_stride is not None else 1
        if K < 1:
            K = 1
        if stride_mode not in {"hold", "sparse"}:
            raise ValueError(f"Unknown stride_mode: {stride_mode}")
        offset = int(stride_offset) if stride_offset is not None else 0

        # Default: no extra stride mask
        stride_mask = None  # [B,T,1] or None

        # ---- mid-once injection (clean-only) ----
        # If enabled, we OVERRIDE the stride masking so that:
        #   - exactly ONE non-terminal position gets injected (a "pilot"/bootstrap state)
        #   - the last position is ALWAYS injected (to keep supervision aligned)
        # This isolates whether a single mid-state exposure is sufficient to bootstrap state usage.
        if bool(mid_once) and inject_mode == "clean" and T > 0:
            if mid_pos_mode not in {"batch", "sample"}:
                raise ValueError(f"Unknown mid_pos_mode: {mid_pos_mode}")

            keep = torch.zeros((B, T, 1), device=device, dtype=torch.float32)

            # choose one non-terminal position in [0, T-2] if possible
            if T >= 2:
                if mid_pos is not None and int(mid_pos) >= 0:
                    t_mid = int(mid_pos)
                    if t_mid >= T - 1:
                        t_mid = T - 2
                    keep[:, t_mid, 0] = 1.0
                else:
                    if mid_pos_mode == "batch":
                        t_mid = int(torch.randint(low=0, high=T - 1, size=(1,), device=device).item())
                        keep[:, t_mid, 0] = 1.0
                    else:  # sample
                        t_mid = torch.randint(low=0, high=T - 1, size=(B,), device=device)
                        keep[torch.arange(B, device=device), t_mid, 0] = 1.0

            # always inject at the last position
            keep[:, T - 1, 0] = 1.0

            stride_mask = keep  # [B,T,1]

        if inject_mode == "clean" and (not bool(mid_once)) and K > 1 and T > 0:
            if stride_mode == "hold":
                # your original behavior: hold the PRE-state for K steps
                held = state_ids.clone()
                for t0 in range(0, T, K):
                    t1 = min(t0 + K, T)
                    held[:, t0:t1] = state_ids[:, t0:t0 + 1].expand(B, t1 - t0)
                state_ids = held

            elif stride_mode == "sparse":
                # sparse behavior: only inject at every K-th position (phase controlled by offset)
                t = torch.arange(T, device=device)  # [T]
                # ---- random phase-shift (TRAIN-TIME) ----
                # We randomize the phase of the sparse mask to encourage phase-invariant usage.
                # This does NOT change state_ids; it only changes which positions are injected (mask=1).
                if random_phase_shift:
                    if phase_shift_mode == "batch":
                        # one shift for the entire batch
                        shift = int(torch.randint(low=0, high=K, size=(1,), device=device).item())
                        eff_offset = offset + shift
                        keep = ((t - eff_offset) % K == 0).to(torch.float32)  # [T]
                        # ---- FORCE LAST STEP VISIBILITY ----
                        keep[T - 1] = 1.0
                        stride_mask = keep.view(1, T, 1).expand(B, T, 1)  # [B,T,1]
                    elif phase_shift_mode == "sample":
                        # independent shift per sample
                        shift = torch.randint(low=0, high=K, size=(B, 1, 1), device=device)  # [B,1,1]
                        tt = t.view(1, T, 1).expand(B, T, 1)  # [B,T,1]
                        keep = (((tt - (offset + shift)) % K) == 0).to(torch.float32)  # [B,T,1]
                        # ---- FORCE LAST STEP VISIBILITY ----
                        keep[T - 1] = 1.0
                        stride_mask = keep
                    else:
                        raise ValueError(f"Unknown phase_shift_mode: {phase_shift_mode}")
                else:
                    # deterministic phase
                    keep = ((t - offset) % K == 0).to(torch.float32)  # [T]
                    stride_mask = keep.view(1, T, 1).expand(B, T, 1)  # [B,T,1]

        # ---- embed/project ----
        s = self.state_proj(self.state_emb(state_ids))  # [B,T,H]

        # ---- IMPORTANT: zero-mask non-injected positions (semantic fix) ----
        # clean: inject all positions (then optionally apply stride_mask for sparse)
        # final/prev: inject only last position
        # none: inject nothing
        if T == 0:
            mask = torch.zeros((B, T, 1), device=device, dtype=s.dtype)
        else:
            mask = torch.zeros((B, T, 1), device=device, dtype=s.dtype)
            if inject_mode == "clean":
                mask[:] = 1.0
            elif inject_mode in {"final", "prev"}:
                mask[:, -1, :] = 1.0
            elif inject_mode == "none":
                pass

        # Apply sparse stride mask (if any): this makes "missing steps" a true no-op
        if stride_mask is not None:
            mask = mask * stride_mask.to(dtype=mask.dtype)

        s = s * mask  # zero means true no-op; prevents identity-bias artifacts

        # ---- run backbone ----
        x = self.tok_emb(input_ids)  # [B,T,H]

        # cache for fusion if requested
        if inject_style in {"fusion", "both"} and inject_mode != "none":
            # If mask is all zeros (shouldn't happen except none), skip.
            if mask.sum().item() > 0:
                self._cached_s = s
                self._cached_gate_zero = bool(gate_zero)
            else:
                self._cached_s = None
                self._cached_gate_zero = False
        else:
            self._cached_s = None
            self._cached_gate_zero = False

        if inject_style in {"input_add", "both"}:
            x = x + s

        out = self.gpt2(
            inputs_embeds=x,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )

        # clear cache
        self._cached_s = None
        self._cached_gate_zero = False

        h_last = out.last_hidden_state[:, -1]
        logits = self.head(h_last)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss
