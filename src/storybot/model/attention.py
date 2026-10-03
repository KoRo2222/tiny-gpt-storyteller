from __future__ import annotations

import math

import torch
from torch import nn

from .rope import RotaryEmbedding

# Cached keys and values of one layer, each (batch, n_heads, past_len, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary position embedding.

    Queries and keys are rotated by RoPE before the dot product, so scores
    depend on relative position and no learned position table is needed.
    An optional KV cache lets generation process only the newest token(s):
    the new tokens' positions start right after the cached ones.
    """

    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        batch, seq_len, d_model = x.shape
        q, k, v = (
            t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(d_model, dim=-1)
        )  # each (batch, n_heads, seq_len, head_dim)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Query i sits at absolute position past_len + i and may attend to
        # keys at positions <= that, i.e. never to the future.
        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
        k_pos = torch.arange(past_len + seq_len, device=x.device)
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
        attn = scores.softmax(dim=-1)

        y = (attn @ v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(y), (k, v)
