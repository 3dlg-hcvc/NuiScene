################################################################################################################################################
"""
Modifed From: https://github.com/wyysf-98/CraftsMan
"""
################################################################################################################################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.silu = mp_silu

    def forward(self, x):
        return self.c_proj(self.silu(self.c_fc(x)))

class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads, qkv_norm):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, qkv_norm=qkv_norm)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x

class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads, n_ctx, qkv_norm):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.qkv_norm = qkv_norm

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.qkv_norm:
            q = normalize(q.permute(0, 2, 1, 3), dim=-1)
            k = normalize(k.permute(0, 2, 1, 3), dim=-1)
            v = normalize(v.permute(0, 2, 1, 3), dim=-1)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_ctx, width, heads, qkv_norm):
        super().__init__()

        self.attn = MultiheadAttention(n_ctx=n_ctx, width=width, heads=heads, qkv_norm=qkv_norm)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return self._forward(x)

class MultiheadCrossAttention(nn.Module):
    def __init__(self, width, heads, qkv_norm, data_width=None):
        super().__init__()
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width)
        self.c_kv = nn.Linear(self.data_width, width * 2)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, qkv_norm=qkv_norm)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x

class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, heads, qkv_norm):
        super().__init__()
        self.heads = heads
        self.qkv_norm = qkv_norm

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2

        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        if self.qkv_norm:
            q = normalize(q.permute(0, 2, 1, 3), dim=-1)
            k = normalize(k.permute(0, 2, 1, 3), dim=-1)
            v = normalize(v.permute(0, 2, 1, 3), dim=-1)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)

        return out

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, width, heads, data_width, qkv_norm):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(width=width, heads=heads, qkv_norm=qkv_norm, data_width=data_width)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x
