import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List


class FrozenDistilGPT2Tagger(nn.Module):
    def __init__(self, num_tokens=60, model_name="distilgpt2", unfreeze_last_n: int = 2):
        super().__init__()
        self.gpt2 = AutoModel.from_pretrained(model_name)
        for p in self.gpt2.parameters():
            p.requires_grad = False

        base = self.gpt2.transformer if hasattr(self.gpt2, "transformer") else self.gpt2

        if unfreeze_last_n > 0:
            for block in base.h[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

        for p in base.ln_f.parameters():
            p.requires_grad = True

        hidden = self.gpt2.config.n_embd
        self.embed = nn.Embedding(num_tokens, hidden)
        self.head = nn.Linear(hidden, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def unfrozen_parameters(self) -> List[torch.nn.Parameter]:
        """
        返回所有 requires_grad=True 且属于 distilgpt2 的参数（不包含 embed/head）
        """
        return [p for p in self.gpt2.parameters() if p.requires_grad]

    def forward(self, input_ids, attention_mask=None, labels=None):
        inputs_embeds = self.embed(input_ids)
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss
