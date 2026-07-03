import torch
import torch.nn.functional as F
import math
from transformer.model import scaled_dot_product_attention

def test_attention_output_shape():
    batch, seq_len, d_k = 2, 5, 8
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (batch, seq_len, d_k)

def test_causal_mask_blocks_future():
    batch, seq_len, d_k = 1, 5, 4
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v1 = torch.randn(batch, seq_len, d_k)
    v2 = v1.clone()
    v2[:, 4, :] = torch.randn(batch, d_k)
    out1 = scaled_dot_product_attention(q, k, v1, causal=True)
    out2 = scaled_dot_product_attention(q, k, v2, causal=True)
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :])

def test_attention_weights_sum_to_one():
    batch, seq_len, d_k = 1, 4, 4
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)
    d_k_local = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k_local)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-6)