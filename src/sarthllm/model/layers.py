import math

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps= 1e-5):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, hidden_states):
        hidden_fp32 = hidden_states.float() # mean and variance in float32 so fp16 does not overflow
        mean = hidden_fp32.mean(dim=-1, keepdim=True)
        variance = hidden_fp32.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (hidden_fp32 - mean) / torch.sqrt(variance + self.eps)
        return (self.weight * normalized + self.bias).to(hidden_states.dtype)    

class GELU(nn.Module):
    def forward(self, x):
        # tanh
        cubic = 0.044715 * x.pow(3)
        tanh_input = math.sqrt(2.0 / math.pi) * (x + cubic)
        return 0.5 * x * (1.0 + torch.tanh(tanh_input))    

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_multiplier=4):
        super().__init__()
        hidden_dim = hidden_multiplier *embedding_dim 
        # notes dekho for reference of this.
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)   

class Residual(nn.Module):
    def __init___(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, hidden_states):
        return hidden_states + self.sublayer(hidden_states)        