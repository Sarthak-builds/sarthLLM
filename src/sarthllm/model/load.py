import torch
from huggingface_hub import hf_hub_download

from sarthllm.model.config import cfg
from sarthllm.model.llm import SarthLLM


def load_model(repo_id, device, filename="best.pt"):
    weights_path = hf_hub_download(repo_id, filename, repo_type="model")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = SarthLLM(cfg)
    model.load_state_dict(checkpoint["model"])
    print(f"loaded {filename} from step {checkpoint['step']}")
    return model.to(device).eval()