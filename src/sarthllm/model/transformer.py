from torch import nn

from sarthllm.model.layers import FeedForward, LayerNorm, Residual
from sarthllm.model.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        dropout_rate = cfg["drop_rate"]
        n_heads = cfg["n_heads"]
        qkv_bias = cfg["qkv_bias"]
        self.layers = nn.Sequential(
            Residual(nn.Sequential(  
                LayerNorm(emb_dim),
                MultiHeadAttention(emb_dim,n_heads, dropout_rate, qkv_bias),
                nn.Dropout(dropout_rate),
            )),
            Residual(nn.Sequential(  
                LayerNorm(emb_dim),
                FeedForward(emb_dim),
                nn.Dropout(dropout_rate),
            )),
        )

    def forward(self, hidden_states):
        return self.layers(hidden_states)    