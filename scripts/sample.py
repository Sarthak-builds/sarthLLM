# Generates text from the trained base model so you can see what it learned.
import argparse
from pathlib import Path

import torch
import yaml

from sarthllm.model.generate import generate_text
from sarthllm.model.load import load_model

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In 1969, the first humans",
    "To make a good cup of tea, you should",
    "The main difference between a virus and a bacterium is",
]


def main(prompt, max_new_tokens, temperature, top_k, seed):
    tc = yaml.safe_load(Path("configs/train.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(tc["hf_runs_repo"], device)
    torch.manual_seed(seed)
    for text in [prompt] if prompt else PROMPTS:
        print(f"\nPROMPT: {text}")
        print("greedy :", generate_text(model, text, max_new_tokens, temperature=0.0))
        print("sampled:", generate_text(model, text, max_new_tokens, temperature, top_k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=None, help="one prompt instead of the built-in list")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2104)
    args = parser.parse_args()
    main(args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed)