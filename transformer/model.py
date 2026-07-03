import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
        q: torch.Tensor, # (batch, seq_len, d_k)
        k: torch.Tensor, # (batch, seq_len, d_k)
        v: torch.Tensor, # (batch, seq_len, d_k)
        causal: bool = True
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k) # @ -> batched matmul
    if causal:
        seq_len = q.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ v

class MultiHeadAttention(nn.Module()):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        #q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output = scaled_dot_product_attention(q, k, v, causal=causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        output = self.W_o(attn_output)
        return output