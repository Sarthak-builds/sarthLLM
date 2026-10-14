# HellaSwag evaluation: pick the most likely ending for each example (random guessing = 25%).
import json
import urllib.request
from pathlib import Path

import torch

from sarthllm.data.tokenizer import encode

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def load_examples(path, limit=None):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HELLASWAG_URL, path)
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    return examples[:limit] if limit else examples


@torch.no_grad()
def score_endings(model, example, device):
    # one row per ending: context + ending, right-padded (padding cannot affect earlier tokens)
    context_ids = encode(example["ctx"])
    ending_ids_list = [encode(" " + ending) for ending in example["endings"]]
    sequences = [context_ids + ending_ids for ending_ids in ending_ids_list]
    input_ids = torch.zeros(len(sequences), max(len(s) for s in sequences), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)

    log_probs = torch.log_softmax(model(input_ids.to(device)).float(), dim=-1)  # [4, tokens, vocab]
    total_scores, mean_scores = [], []
    start = len(context_ids)
    for row, ending_ids in enumerate(ending_ids_list):
        positions = torch.arange(start - 1, start - 1 + len(ending_ids), device=device)  # position t predicts token t+1
        targets = torch.tensor(ending_ids, device=device)
        token_log_probs = log_probs[row, positions, targets]
        total_scores.append(token_log_probs.sum().item())
        mean_scores.append(token_log_probs.mean().item())  # per-token average: fair to long endings
    return total_scores, mean_scores


def evaluate_hellaswag(model, examples, device):
    correct, correct_norm = 0, 0
    for example in examples:
        total_scores, mean_scores = score_endings(model, example, device)
        label = int(example["label"])
        correct += int(max(range(4), key=total_scores.__getitem__) == label)
        correct_norm += int(max(range(4), key=mean_scores.__getitem__) == label)
    n = len(examples)
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n_examples": n}