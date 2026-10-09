import torch
from torch import nn

from sarthllm.model.layers import LayerNorm
from sarthllm.model.transformer import TransformerBlock


class SarthLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.out_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        _, num_tokens = input_ids.shape
        if num_tokens > self.context_length:
            raise ValueError(f"got {num_tokens} tokens, max is {self.context_length}")

        token_embeds = self.tok_emb(input_ids)  
        positions = torch.arange(num_tokens, device=input_ids.device)
        position_embeds = self.pos_emb(positions)  

        hidden_states = self.drop_emb(token_embeds + position_embeds)
        hidden_states = self.trf_blocks(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.out_head(hidden_states)
        return logits

