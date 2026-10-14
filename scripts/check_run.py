# Shows how far training has got, using the checkpoint name and metrics stored in the HF runs repo.
import json

from huggingface_hub import HfApi, hf_hub_download

REPO = "Sarthakhq/sarthllm-runs"

files = HfApi().list_repo_files(REPO, repo_type="model")
checkpoints = sorted(f for f in files if f.startswith("ckpt_"))
print("files:", files)
print("newest checkpoint:", checkpoints[-1] if checkpoints else None)

metrics_path = hf_hub_download(REPO, "metrics.jsonl", repo_type="model")
records = [json.loads(line) for line in open(metrics_path)]
last = records[-1]
evals = [r for r in records if "test_loss" in r]
print(f"last step {last['step']} | train loss {last['train_loss']:.3f} | tokens seen {last['tokens_seen']:,}")
if evals:
    print(f"last eval: step {evals[-1]['step']} | test loss {evals[-1]['test_loss']:.3f}")
speeds = [r["tokens_per_sec"] for r in records[-50:]]
print(f"recent speed: {sum(speeds) / len(speeds):,.0f} tok/s")