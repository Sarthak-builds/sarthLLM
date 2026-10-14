# Evaluates the trained base model: held-out test loss and HellaSwag. Writes the results to a JSON file.
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from sarthllm.data.dataset import Sarthllm_Dataset, create_dataloader, train_test_split
from sarthllm.data.shards import TokenShards
from sarthllm.eval.hellaswag import evaluate_hellaswag, load_examples
from sarthllm.model.config import cfg
from sarthllm.model.load import load_model
from sarthllm.train.loss import calc_loss_loader


def test_loss(model, tc, device, max_windows, batch_size):
    shards = TokenShards(tc["data_dir"], max_tokens=tc["max_tokens"])
    dataset = Sarthllm_Dataset(shards, max_length=cfg["context_length"])
    _, test_ds = train_test_split(dataset, tc["test_ratio"], tc["seed"])  # windows training never saw
    num_windows = min(max_windows, len(test_ds)) if max_windows else len(test_ds)
    loader = create_dataloader(Subset(test_ds, range(num_windows)), batch_size, shuffle=False)
    return calc_loss_loader(loader, model, device), num_windows


def main(test_windows, hellaswag_limit, batch_size, output):
    tc = yaml.safe_load(Path("configs/train.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(tc["hf_runs_repo"], device)

    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        loss, num_windows = test_loss(model, tc, device, test_windows, batch_size)
        print(f"test loss {loss:.3f} | perplexity {math.exp(loss):.1f} | {num_windows} windows")
        examples = load_examples("artifacts/eval/hellaswag_val.jsonl", hellaswag_limit or None)
        hellaswag = evaluate_hellaswag(model, examples, device)

    p, n = hellaswag["acc_norm"], hellaswag["n_examples"]
    stderr = math.sqrt(p * (1 - p) / n)
    print(f"HellaSwag acc {hellaswag['acc']:.3f} | acc_norm {p:.3f} (+/- {stderr:.3f}) | "
          f"{n} examples | random = 0.250")

    results = {"test_loss": loss, "perplexity": math.exp(loss), "test_windows": num_windows,
               "hellaswag_acc": hellaswag["acc"], "hellaswag_acc_norm": p,
               "hellaswag_stderr": stderr, "hellaswag_examples": n}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-windows", type=int, default=200, help="0 = the whole test split")
    parser.add_argument("--hellaswag-limit", type=int, default=500, help="0 = all 10,042 examples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="artifacts/eval/base_results.json")
    args = parser.parse_args()
    main(args.test_windows, args.hellaswag_limit, args.batch_size, args.output)