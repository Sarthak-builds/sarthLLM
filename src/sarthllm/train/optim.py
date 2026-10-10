# Optimizer (AdamW, weight decay jo ki sirf bade matrices pr lagega) and learning-rate schedule (warmup aur cosine wala concept).
import math
import torch


def build_optimizer(model, peak_lr, weight_decay, betas=(0.9, 0.95), fused=False):
    params = [p for p in model.parameters() if p.requires_grad]  
    decay_params = [p for p in params if p.dim() >= 2]  
    no_decay_params = [p for p in params if p.dim() < 2]  
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=peak_lr, betas=betas, fused=fused)


def get_lr(step, peak_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps: 
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps: 
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps) 
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  
    return min_lr + cosine * (peak_lr - min_lr)  


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr