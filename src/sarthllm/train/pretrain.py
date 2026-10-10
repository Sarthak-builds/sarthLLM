# Pretraining loop: accumulation, fp16 autocast, LR schedule, evaluation, sampling, resumable checkpoints.
import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from sarthllm.data.dataset import Sarthllm_Dataset, create_dataloader, train_test_split
from sarthllm.data.shards import TokenShards
from sarthllm.model.config import cfg
from sarthllm.model.generate import generate_text
from sarthllm.model.llm import SarthLLM
from sarthllm.train.checkpoint import load_latest_checkpoint, save_best, save_checkpoint
from sarthllm.train.loss import calc_loss_batch, calc_loss_loader
from sarthllm.train.optim import build_optimizer, get_lr, set_lr

# settings that must not change between a run and its resume: they define the schedule and data order
LOCKED_KEYS = ["max_tokens", "test_ratio", "tokens_per_step", "max_steps", "peak_lr", "min_lr",
               "warmup_steps", "weight_decay", "seed"]


def load_train_config(config_path, overrides):
    train_cfg = yaml.safe_load(Path(config_path).read_text())
    for item in overrides:  # e.g. --set max_steps=20 micro_batch_size=2
        key, value = item.split("=", 1)
        if key not in train_cfg:
            raise SystemExit(f"unknown setting '{key}': it is not in {config_path}")
        value = yaml.safe_load(value)
        default = train_cfg[key]
        if isinstance(value, str) and isinstance(default, (int, float)):
            value = float(value)  # PyYAML reads 3e-4 and 2e7 as strings: make them numbers
            if isinstance(default, int):
                value = int(value)
        train_cfg[key] = value
    return train_cfg


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def amp_context(device, enabled):
    # fp16 autocast: matmuls run in half precision, the rest stays float32
    return torch.autocast(device_type=device, dtype=torch.float16, enabled=enabled)


def make_train_loader(train_ds, micro_batch_size, sequences_done, seed):
    # one fixed shuffled order; resuming just skips the sequences already trained on
    order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(seed)).tolist()
    return create_dataloader(Subset(train_ds, order[sequences_done:]), micro_batch_size, shuffle=False)


def check_resume_compatible(checkpoint, tc):
    if checkpoint["model_cfg"] != cfg:
        raise SystemExit("checkpoint was trained with a different model config")
    for key in LOCKED_KEYS:
        if checkpoint["train_cfg"].get(key) != tc.get(key):
            raise SystemExit(
                f"cannot resume: '{key}' changed ({checkpoint['train_cfg'].get(key)} -> {tc.get(key)}). "
                "Revert it, or use a new out_dir to start a fresh run."
            )


def main(config_path, overrides):
    tc = load_train_config(config_path, overrides)
    torch.manual_seed(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"  # fp16 + gradient scaler (the T4 has no bf16)
    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    # data: shards -> windows -> train/test split
    shards = TokenShards(tc["data_dir"], max_tokens=tc["max_tokens"])
    dataset = Sarthllm_Dataset(shards, max_length=cfg["context_length"])
    train_ds, test_ds = train_test_split(dataset, tc["test_ratio"], tc["seed"])
    test_loader = create_dataloader(test_ds, tc["micro_batch_size"], shuffle=False)

    # batch math: micro-batches are accumulated until one optimizer step has tokens_per_step tokens
    sequences_per_step = tc["tokens_per_step"] // cfg["context_length"]
    if sequences_per_step % tc["micro_batch_size"] != 0:
        raise ValueError("tokens_per_step / context_length must be divisible by micro_batch_size")
    grad_accum_steps = sequences_per_step // tc["micro_batch_size"]
    tokens_per_step = sequences_per_step * cfg["context_length"]
    total_steps = len(train_ds) // sequences_per_step  # one pass over the training windows
    if tc.get("max_steps"):
        total_steps = min(total_steps, tc["max_steps"])
    if total_steps == 0:
        raise ValueError("not enough training data for a single optimizer step")

    # model, optimizer (built after .to(device)), gradient scaler
    model = SarthLLM(cfg).to(device)
    optimizer = build_optimizer(
        model, tc["peak_lr"], tc["weight_decay"], (tc["beta1"], tc["beta2"]), fused=use_amp
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # resume from the newest checkpoint, if there is one
    start_step, tokens_seen, best_test_loss = 0, 0, float("inf")
    checkpoint = load_latest_checkpoint(out_dir, device)
    if checkpoint is not None:
        check_resume_compatible(checkpoint, tc)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
        tokens_seen = checkpoint["tokens_seen"]
        best_test_loss = checkpoint["best_test_loss"]
        del checkpoint
        print(f"resumed from step {start_step}")
    if start_step >= total_steps:
        print("training already finished")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | params {num_params:,} | {tokens_per_step:,} tokens/step | "
          f"accumulation {grad_accum_steps} | steps {start_step:,} -> {total_steps:,}")

    train_iter = iter(make_train_loader(
        train_ds, tc["micro_batch_size"], start_step * sequences_per_step, tc["seed"]
    ))

    def checkpoint_state(completed_steps):
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "step": completed_steps, "tokens_seen": tokens_seen,
            "best_test_loss": best_test_loss, "model_cfg": cfg, "train_cfg": tc,
        }

    completed_steps = start_step
    last_save_time = time.monotonic()
    model.train()
    try:
        for step in range(start_step, total_steps):
            step_start = time.perf_counter()
            lr = get_lr(step, tc["peak_lr"], tc["min_lr"], tc["warmup_steps"], total_steps)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for _ in range(grad_accum_steps):
                input_ids, target_ids = next(train_iter)
                with amp_context(device, use_amp):
                    loss = calc_loss_batch(input_ids, target_ids, model, device)
                scaler.scale(loss / grad_accum_steps).backward()  # gradients add up across micro-batches
                step_loss += loss.detach() / grad_accum_steps

            scaler.unscale_(optimizer)  # so clipping sees the true gradient size
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            completed_steps = step + 1  # the weights just changed: count the step before anything can interrupt
            tokens_seen += tokens_per_step

            train_loss = step_loss.item()  # also waits for the GPU, so the timing below is honest
            if not math.isfinite(train_loss):
                raise SystemExit(f"loss is {train_loss} at step {completed_steps}: stopping")
            record = {
                "step": completed_steps, "tokens_seen": tokens_seen, "lr": lr,
                "train_loss": train_loss, "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_per_step / (time.perf_counter() - step_start),
            }
            if device == "cuda":
                record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

            is_eval_step = completed_steps % tc["eval_every"] == 0 or completed_steps == total_steps
            if is_eval_step:
                with amp_context(device, use_amp):
                    record["test_loss"] = calc_loss_loader(test_loader, model, device, tc["eval_batches"])
                    sample = generate_text(model, tc["sample_prompt"], max_new_tokens=40)
                if record["test_loss"] < best_test_loss:
                    best_test_loss = record["test_loss"]
                    save_best(model, out_dir, completed_steps, best_test_loss)

            if completed_steps % tc["log_every"] == 0 or is_eval_step:
                append_jsonl(metrics_path, record)
                print(f"step {completed_steps:>5}/{total_steps} | loss {train_loss:.3f} | lr {lr:.2e} | "
                      f"grad {record['grad_norm']:.2f} | {record['tokens_per_sec']:,.0f} tok/s")
            if is_eval_step:
                print(f"  >> test loss {record['test_loss']:.3f} | sample: {sample!r}")

            minutes_since_save = (time.monotonic() - last_save_time) / 60
            if minutes_since_save > tc["checkpoint_every_minutes"] or completed_steps == total_steps:
                save_checkpoint(checkpoint_state(completed_steps), out_dir, completed_steps)
                last_save_time = time.monotonic()
    except KeyboardInterrupt:
        print(f"interrupted: saving checkpoint at step {completed_steps}")
        save_checkpoint(checkpoint_state(completed_steps), out_dir, completed_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    main(args.config, args.set)