import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_rate, qkv_bias=False):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout_rate = dropout_rate

        self.W_query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_key = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_value = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def _split_heads(self, projected):
        batch_size, num_tokens, _ = projected.shape
        split = projected.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return split.transpose(1, 2)

    def _merge_heads(self, per_head_context):
        batch_size, _, num_tokens, _ = per_head_context.shape
        merged = per_head_context.transpose(1, 2)
        return merged.reshape(batch_size, num_tokens, self.num_heads * self.head_dim)

    def forward(self, hidden_states):
        queries = self._split_heads(self.W_query(hidden_states))
        keys = self._split_heads(self.W_key(hidden_states))
        values = self._split_heads(self.W_value(hidden_states))

        attn_dropout_p = self.dropout_rate if self.training else 0.0  
        per_head_context = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True, dropout_p=attn_dropout_p
        )
        contextual_embeds = self._merge_heads(per_head_context)  
        return self.out_proj(contextual_embeds)