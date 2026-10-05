import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, drop_rate, qkv_bias=False):
        super.__init__()
        if emb_dim % n_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.drop_rate = drop_rate
        
        self.W_query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_key = nn.Linear(emb_dim, emb_dim, bias = qkv_bias)
        self.W_value = nn.Linear(emb_dim, emb_dim, bias= qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def _split_heads(self, x):
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, self.n_heads * self.head_dim)
    
    def forward(self, x):
        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        dropout_p = self.drop_rate if self.training else 0.0  
        head_outputs = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True, dropout_p=dropout_p
        )
        contextual_embeds = self._merge_heads(head_outputs)  # [batch, tokens, emb_dim]
        return self.out_proj(contextual_embeds)
        

