import torch
from sarthllm.data.tokenizer import EOT_ID, decode, encode

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.context_length :] 
        next_token_logits = model(context)[:, -1, :]  

        if top_k is not None: 
            top_scores, _ = torch.topk(next_token_logits, top_k)
            kth_best = top_scores[:, -1:]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < kth_best, float("-inf")
            )

        if temperature > 0.0:  
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:  
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if eos_id is not None and (next_id == eos_id).all():  
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    model.train(was_training)  
    return input_ids


def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    device = next(model.parameters()).device
    input_ids = torch.tensor([encode(prompt)], device=device)
    output_ids = generate(model, input_ids, max_new_tokens, temperature, top_k, eos_id=EOT_ID)
    return decode(output_ids[0].tolist())