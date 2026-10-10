# Checkpoint helpers for resumable training: atomic saves, keep the newest few, track the best model.
import os
from pathlib import Path

import torch


def _atomic_save(obj, path):
    tmp_path = Path(str(path) + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path) 


def list_checkpoints(out_dir):
    return sorted(Path(out_dir).glob("ckpt_*.pt"))  


def save_checkpoint(state, out_dir, step, keep_last=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(state, out_dir / f"ckpt_{step:06d}.pt")
    for old_checkpoint in list_checkpoints(out_dir)[:-keep_last]:
        old_checkpoint.unlink()


def save_best(model, out_dir, step, test_loss):
    state = {"model": model.state_dict(), "step": step, "test_loss": test_loss}
    _atomic_save(state, Path(out_dir) / "best.pt")


def load_latest_checkpoint(out_dir, device):
    checkpoints = list_checkpoints(out_dir)
    if not checkpoints:
        return None
    return torch.load(checkpoints[-1], map_location=device, weights_only=True)