import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Baselines (unchanged)
# -------------------------

class BaselineAdapter(nn.Module):
    """
    非顺序 baseline（token-wise MLP + pooling），保留用于对比“没法利用顺序/状态”。
    """

    def __init__(
            self,
            num_tokens: int = 60,
            d_model: int = 64,
            mlp_layers: int = 2,
            pool: str = "mean",
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.pool = pool

        layers = []
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)  # [B,T,D]
        x = self.mlp(x)  # [B,T,D]

        if self.pool == "mean":
            v = x.mean(dim=1)
        elif self.pool == "last":
            v = x[:, -1]
        else:
            raise ValueError(f"unknown pool: {self.pool}")

        logits = self.head(v)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss


class GRUBaseline(nn.Module):
    """
    合理顺序 baseline：Embedding -> GRU -> last hidden -> Linear -> 60
    目标：能学 train_len=64，但长度外推通常会掉（作为对照）。
    """

    def __init__(
            self,
            num_tokens: int = 60,
            d_model: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)  # [B,T,D]
        _, h_n = self.gru(x)  # h_n: [L,B,D]
        h_last = h_n[-1]  # [B,D]
        logits = self.head(h_last)  # [B,60]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss


class StateScanPluginP0(nn.Module):
    """
    P0 plugin sanity：
      - 仅 Matrix lookup + scan + readout
      - 这里仍然是“学 60 个矩阵”，用于 sanity / 对照
    """

    def __init__(
            self,
            num_tokens: int = 60,
            d_model: int = 16,
            readout_hidden: int | None = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.repr = nn.Parameter(torch.randn(num_tokens, d_model, d_model) * 0.02)

        if readout_hidden is None:
            self.readout = nn.Linear(d_model * d_model, num_tokens)
        else:
            self.readout = nn.Sequential(
                nn.Linear(d_model * d_model, readout_hidden),
                nn.ReLU(),
                nn.Linear(readout_hidden, num_tokens),
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape
        d = self.d_model

        M = self.repr[input_ids]  # [B,T,d,d]

        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=input_ids.device)
            M = M[:, perm]

        if no_scan:
            S = M[:, -1]
        else:
            eye = torch.eye(d, device=input_ids.device).unsqueeze(0).expand(B, d, d).contiguous()
            S = eye
            for t in range(T):
                if reset_each_step:
                    S = eye
                S = torch.bmm(M[:, t], S)

        logits = self.readout(S.reshape(B, -1))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss


class SelectiveScanPlugin(nn.Module):
    """
    Barrington-inspired（轻量版，仍在学矩阵）：
      - token -> softmax over K
      - generators: G_k 是可学习参数（会回到“学表示矩阵”的问题）
    """

    def __init__(
            self,
            num_tokens: int = 60,
            d_model: int = 16,
            k_generators: int = 16,
            temp: float = 1.0,
            gumbel_hard: bool = False,
            readout_hidden: int | None = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k_generators
        self.temp = temp
        self.gumbel_hard = gumbel_hard

        self.tok_emb = nn.Embedding(num_tokens, d_model)
        self.tok_to_k = nn.Linear(d_model, k_generators)

        self.generators = nn.Parameter(torch.randn(k_generators, d_model, d_model) * 0.02)

        if readout_hidden is None:
            self.readout = nn.Linear(d_model * d_model, num_tokens)
        else:
            self.readout = nn.Sequential(
                nn.Linear(d_model * d_model, readout_hidden),
                nn.ReLU(),
                nn.Linear(readout_hidden, num_tokens),
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape

        x = self.tok_emb(input_ids)  # [B,T,D]
        logits_k = self.tok_to_k(x)  # [B,T,K]

        if self.gumbel_hard:
            w = F.gumbel_softmax(logits_k, tau=self.temp, hard=True, dim=-1)
        else:
            w = F.softmax(logits_k / max(self.temp, 1e-6), dim=-1)

        M = torch.einsum("btk,kij->btij", w, self.generators)

        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=input_ids.device)
            M = M[:, perm]

        if no_scan:
            S = M[:, -1]
        else:
            d = self.d_model
            eye = torch.eye(d, device=input_ids.device).unsqueeze(0).expand(B, d, d).contiguous()
            S = eye
            for t in range(T):
                if reset_each_step:
                    S = eye
                S = torch.bmm(M[:, t], S)

        logits = self.readout(S.reshape(B, -1))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss


# -------------------------
# Exact executor (unchanged)
# -------------------------

class A5ExactScanPlugin(nn.Module):
    """
    ✅ 最稳“固定基底”插件：直接用 Cayley table 做显式状态更新（不学矩阵、不学基）
    状态 s_t 是群元素 id (0..59)，更新：
        s_t = x_t ∘ s_{t-1}
    输出 logits 直接 one-hot(s_T)
    """

    def __init__(self, mul_table, id_id: int, num_tokens: int = 60):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.to(dtype=torch.long)

        assert mul_table.dim() == 2
        assert mul_table.size(0) == num_tokens and mul_table.size(1) == num_tokens
        self.register_buffer("mul", mul_table)  # [60,60]
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        x = input_ids
        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=device)
            x = x[:, perm]

        if no_scan:
            s = x[:, -1]
        else:
            s = torch.full((B,), self.id_id, device=device, dtype=torch.long)
            for t in range(T):
                if reset_each_step:
                    s = torch.full((B,), self.id_id, device=device, dtype=torch.long)
                s = self.mul[x[:, t], s]

        logits = torch.full((B, self.num_tokens), -10.0, device=device)
        logits.scatter_(1, s.view(-1, 1), 10.0)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


# -------------------------
# NEW: Learn token -> generator-program (routing)
# -------------------------

def _closure_size(mul: torch.Tensor, id_id: int, gens: list[int], max_iters: int = 5000) -> int:
    """Return size of subgroup generated by `gens` under left-multiplication closure."""
    seen = set([id_id])
    q = collections.deque([id_id])
    iters = 0
    while q and iters < max_iters:
        iters += 1
        a = q.popleft()
        for g in gens:
            b = int(mul[g, a].item())  # g ∘ a
            if b not in seen:
                seen.add(b)
                q.append(b)
            b2 = int(mul[a, g].item())  # a ∘ g  (helps reach closure faster)
            if b2 not in seen:
                seen.add(b2)
                q.append(b2)
    return len(seen)


def _find_two_generators(mul: torch.Tensor, id_id: int, num_tokens: int = 60, seed: int = 0) -> tuple[int, int]:
    """
    Find a pair (g0,g1) that generates the whole group (closure size == 60).
    Deterministic-ish: shuffles candidates with seed then tries.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    candidates = torch.randperm(num_tokens, generator=g).tolist()
    # skip identity if possible
    if id_id in candidates:
        candidates.remove(id_id)
        candidates = [id_id] + candidates  # keep id at front but we will avoid picking it
    best = (candidates[0], candidates[1], 0)
    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            g0 = candidates[i]
            g1 = candidates[j]
            if g0 == id_id or g1 == id_id:
                continue
            sz = _closure_size(mul, id_id, [g0, g1])
            if sz > best[2]:
                best = (g0, g1, sz)
            if sz == num_tokens:
                return g0, g1
    # fallback: return best even if not full (should almost always find full for A5)
    return best[0], best[1]


def _bfs_shortest_words(mul: torch.Tensor, id_id: int, g0: int, g1: int, L: int, num_tokens: int = 60):
    """
    Compute a shortest generator word (over {g0,g1}) for each element, up to length L.
    Returns: words [num_tokens, L] with entries in {0,1,2} where 2 means PAD/ID.
    """
    # BFS in Cayley graph from identity using right-multiplication by generators
    # We'll build words for elements reachable within L.
    parent = [-1] * num_tokens
    parent_gen = [-1] * num_tokens  # 0 for g0, 1 for g1
    dist = [10 ** 9] * num_tokens

    dist[id_id] = 0
    dq = collections.deque([id_id])

    gens = [(g0, 0), (g1, 1)]
    while dq:
        u = dq.popleft()
        if dist[u] >= L:
            continue
        for g_id, g_idx in gens:
            v = int(mul[g_id, u].item())  # left multiply: v = g ∘ u
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                parent[v] = u
                parent_gen[v] = g_idx
                dq.append(v)

    # reconstruct words
    words = torch.full((num_tokens, L), 2, dtype=torch.long)  # 2=PAD/ID
    reachable = 0
    if True:
        for x in range(num_tokens):
            if x == id_id:
                reachable += 1
                continue
            if dist[x] <= L:
                # backtrack gens
                seq = []
                cur = x
                while cur != id_id and cur != -1:
                    seq.append(parent_gen[cur])
                    cur = parent[cur]
                seq = list(reversed(seq))
                # fill
                for i, bit in enumerate(seq[:L]):
                    words[x, i] = int(bit)
                reachable += 1
    return words, reachable


class A5RouteToGeneratorsPlugin(nn.Module):
    """
    ✅ 学 token -> 指令选择（路由）的稳健版本：
      - 固定执行器：mul Cayley table
      - 固定两个生成元 g0,g1（自动搜索到能生成全群）
      - 对每个 token，学习一段长度 L 的指令串（每位选 g0 / g1 / PAD(=ID)）
      - 执行器把该 token 的指令串 fold 成一个群元素，再对全序列做 fold

    训练：
      - main loss: final-only (y_T)
      - aux loss (默认开): 让每个 token 的“编译结果”= token 本身（密集监督，确保 len=2 也能学起来）
        这不等价于学矩阵表示基，学的是“如何用生成元程序表达该元素”。
    """

    def __init__(
            self,
            mul_table,
            id_id: int,
            num_tokens: int = 60,
            prog_len: int = 8,
            temp: float = 1.0,
            gumbel_hard: bool = True,
            aux_weight: float = 1.0,
            seed: int = 0,
            g0: int | None = None,
            g1: int | None = None,
            use_teacher_words: bool = True,
    ):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul = mul_table.to(dtype=torch.long)
        assert mul.dim() == 2 and mul.size(0) == num_tokens and mul.size(1) == num_tokens
        self.register_buffer("mul", mul)
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

        self.L = int(prog_len)
        self.temp = float(temp)
        self.gumbel_hard = bool(gumbel_hard)
        self.aux_weight = float(aux_weight)
        self.use_teacher_words = bool(use_teacher_words)

        # choose generators
        if g0 is None or g1 is None:
            g0, g1 = _find_two_generators(self.mul, self.id_id, num_tokens=self.num_tokens, seed=seed)
        self.g0 = int(g0)
        self.g1 = int(g1)
        self.register_buffer("g_ids", torch.tensor([self.g0, self.g1, self.id_id], dtype=torch.long))  # 2=PAD/ID

        # Precompute teacher shortest words up to L (optional)
        words, reachable = _bfs_shortest_words(self.mul, self.id_id, self.g0, self.g1, self.L,
                                               num_tokens=self.num_tokens)
        self.register_buffer("teacher_words", words)  # [60,L] in {0,1,2}
        self.reachable = int(reachable)

        # Learnable route logits: token -> [L,3] (choose g0/g1/PAD)
        self.route_logits = nn.Parameter(torch.zeros(self.num_tokens, self.L, 3))

        self.loss_final = nn.CrossEntropyLoss()
        self.loss_route = nn.CrossEntropyLoss(ignore_index=2)  # ignore PAD label when supervising bits

    def _sample_program_bits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B,T] -> bits: [B,T,L] in {0,1,2}
        """
        logits = self.route_logits[token_ids]  # [B,T,L,3]
        if self.gumbel_hard:
            w = F.gumbel_softmax(logits, tau=self.temp, hard=True, dim=-1)
            bits = w.argmax(dim=-1)
        else:
            probs = F.softmax(logits / max(self.temp, 1e-6), dim=-1)
            bits = probs.argmax(dim=-1)
        return bits

    def _exec_program(self, bits: torch.Tensor) -> torch.Tensor:
        """
        bits: [B,T,L] in {0,1,2} -> element ids: [B,T]
        Execute left-multiplication: s <- g_bit ∘ s, with bit=2 meaning ID (no-op).
        """
        B, T, L = bits.shape
        device = bits.device
        s = torch.full((B, T), self.id_id, device=device, dtype=torch.long)
        g_ids = self.g_ids.to(device)  # [3]
        for i in range(L):
            g = g_ids[bits[:, :, i]]  # [B,T]
            s = self.mul[g, s]  # [B,T]
        return s

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        """
        For compatibility with your trainer, we accept the same ablation flags:
        - no_scan: skip folding, only last token compiled element
        - shuffle_M: shuffle token order before folding
        - reset_each_step: sabotage folding (reset state each step)
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = input_ids
        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=device)
            x = x[:, perm]

        if self.training and self.use_teacher_words:
            # 训练时用 teacher words 来执行，保证 final loss 有意义、acc 能起飞
            bits = self.teacher_words[x].to(device)  # [B,T,L] in {0,1,2}
        else:
            # 评测时用路由预测的 bits 来执行，检验是否学到路由
            bits = self._sample_program_bits(x)  # [B,T,L] in {0,1,2}
        elems = self._exec_program(bits)  # [B,T] each token compiled into group element id

        # fold across sequence to get final group element
        if no_scan:
            y = elems[:, -1]
        else:
            y = torch.full((B,), self.id_id, device=device, dtype=torch.long)
            for t in range(T):
                if reset_each_step:
                    y = torch.full((B,), self.id_id, device=device, dtype=torch.long)
                y = self.mul[elems[:, t], y]  # y <- elem_t ∘ y

        logits = torch.zeros((B, self.num_tokens), device=device)
        logits.scatter_(1, y.view(-1, 1), 5.0)

        loss = None
        aux = None
        if labels is not None:
            loss_final = self.loss_final(logits, labels)

            # Optional dense supervision to make routing learnable and stable:
            # compile(token) should equal token itself
            loss_aux = torch.tensor(0.0, device=device)

            if self.aux_weight > 0:
                if self.use_teacher_words:
                    # supervise bits towards precomputed shortest words (strong & stable)
                    tgt = self.teacher_words[x]  # [B,T,L] in {0,1,2}; 2 means PAD (ignored)
                    logits_bits = self.route_logits[x].reshape(-1, 3)  # [B*T*L,3]
                    tgt_bits = tgt.reshape(-1)  # [B*T*L]
                    loss_aux = self.loss_route(logits_bits, tgt_bits)
                else:
                    # if no teacher words, at least encourage low-entropy routing (optional)
                    # (still works with main final loss, but learning is harder)
                    logits_bits = self.route_logits[x]  # [B,T,L,3]
                    probs = F.softmax(logits_bits, dim=-1)
                    entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)  # [B,T,L]
                    loss_aux = entropy.mean()

            loss = loss_final + self.aux_weight * loss_aux
            aux = {"loss_final": float(loss_final.item()), "loss_aux": float(loss_aux.item())}

        return logits, loss


class A5RouteToElementPlugin(nn.Module):
    """
    ✅ 稳定版本：学习 token -> 选择一个群元素 g（60-way routing），再用 exact mul 执行 scan
    - 不学矩阵、不学基
    - 不学多步程序（避免 bit error 放大）
    - final-only 监督也能学（相当于学每个 token 的“动作”）
    """

    def __init__(self, mul_table, id_id: int, num_tokens: int = 60, temp: float = 1.0, aux_weight:float=0.0):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.to(dtype=torch.long)
        self.register_buffer("mul", mul_table)  # [60,60]
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)
        self.temp = float(temp)
        self.aux_weight = float(aux_weight)
        self._aux_weight_override = None

        # learnable routing table: token -> logits over 60 actions
        self.route_logits = nn.Parameter(torch.zeros(num_tokens, num_tokens))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor | None = None,
            no_scan: bool = False,
            shuffle_M: bool = False,
            reset_each_step: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        x = input_ids
        if shuffle_M and T > 1:
            perm = torch.randperm(T, device=device)
            x = x[:, perm]

        # token -> action distribution over group elements
        logits_g = self.route_logits[x]  # [B,T,60]
        probs_g = torch.softmax(logits_g / max(self.temp, 1e-6), dim=-1)  # [B,T,60]

        # Training: use expected-update (soft) to keep gradient signal.
        # But executor is discrete mul-table; we implement soft update by marginalizing:
        # s is a distribution over states: [B,60]
        if no_scan:
            # just use last token's distribution as state distribution
            s_dist = probs_g[:, -1]  # [B,60]
        else:
            s_dist = torch.zeros(B, self.num_tokens, device=device)
            s_dist[:, self.id_id] = 1.0
            for t in range(T):
                if reset_each_step:
                    s_dist.zero_()
                    s_dist[:, self.id_id] = 1.0

                g_dist = probs_g[:, t]  # [B,60]
                # next_s[h] = sum_{g,s} g_dist[g] * s_dist[s] * 1[mul[g,s]==h]
                # implement via gather: for each g, permute s_dist by mul[g, :]
                next_s = torch.zeros_like(s_dist)
                for g in range(self.num_tokens):
                    # h = mul[g, s]
                    h = self.mul[g]  # [60]
                    # add g_dist[:,g] * s_dist shifted by mapping s->h
                    next_s.index_add_(1, h, s_dist * g_dist[:, g].unsqueeze(1))
                s_dist = next_s

        # logits for final class: use log of distribution
        logits = (s_dist.clamp_min(1e-9)).log()

        loss = None
        if labels is not None:
            loss_final = self.loss_fn(logits, labels)

            # ✅ aux routing supervision: encourage router(token)=token (dense & stable)
            # logits_g: [B,T,60], target: [B,T]
            loss_route = F.cross_entropy(
                logits_g.reshape(-1, self.num_tokens),
                x.reshape(-1),
            )

            # you can tune this weight; start big to bootstrap
            w = self.aux_weight
            if hasattr(self, "_aux_weight_override") and self._aux_weight_override is not None:
                w = float(self._aux_weight_override)

            loss = loss_final + w * loss_route
        return logits, loss
