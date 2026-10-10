import torch
import torch.nn.functional as F


def calc_loss_batch(input_ids, target_ids, model, device):
    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
    logits = model(input_ids)  
    return F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())


@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    was_training = model.training
    model.eval()  

    total_loss, batches_seen = 0.0, 0
    for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        total_loss += calc_loss_batch(input_ids, target_ids, model, device).item()
        batches_seen += 1

    model.train(was_training)  
    return total_loss / batches_seen if batches_seen else float("nan")