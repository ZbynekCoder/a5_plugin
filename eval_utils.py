import torch


@torch.no_grad()
def compute_metrics(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100

    correct = (preds == labels) & mask
    token_correct = correct.sum().item()
    token_total = mask.sum().item()
    token_acc = token_correct / max(token_total, 1)

    per_pos_ok = ((preds == labels) | (~mask)).all(dim=1)
    full_seq_acc = per_pos_ok.float().mean().item()

    return token_acc, full_seq_acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_token_correct = 0
    total_token = 0
    total_seq_correct = 0
    total_seq = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits, _ = model(input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)
        mask = labels != -100

        correct = (preds == labels) & mask
        total_token_correct += correct.sum().item()
        total_token += mask.sum().item()

        per_pos_ok = ((preds == labels) | (~mask)).all(dim=1)
        total_seq_correct += per_pos_ok.sum().item()
        total_seq += labels.size(0)

    token_acc = total_token_correct / max(total_token, 1)
    full_seq_acc = total_seq_correct / max(total_seq, 1)
    return token_acc, full_seq_acc
