"""Shared modules: Conformer blocks, multi-head attention, convolution modules."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth — randomly drops entire residual branches during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=8000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, T, head_dim]

        # Flash / memory-efficient attention: does NOT materialize the [B,H,T,T]
        # score matrix → fast + low VRAM even at frame-level T=500.
        attn_mask = mask[:, None, None, :] if mask is not None else None  # [B,1,1,T] bool, True=attend
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, dim * 2, 1)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pw_conv2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = F.glu(x, dim=1)
        x = self.dw_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    """Conformer block: FF → MHA → Conv → FF (Macaron-style).

    use_conv=False drops the depthwise ConvolutionModule, giving a pure-attention
    Macaron-transformer block (FF → MHA → FF). The conv is memory-bandwidth-bound
    (depthwise, poor GPU utilization) and dominates frame-level cost at T=500; the
    decoder only needs the (cheap) global attention, so it sets use_conv=False.
    """

    def __init__(self, dim, num_heads, ff_mult=4, conv_kernel=31, dropout=0.1,
                 drop_path=0.0, use_conv=True):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_mult, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel, dropout) if use_conv else None
        self.ff2 = FeedForward(dim, ff_mult, dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.drop_path(0.5 * self.ff1(x))
        x = x + self.drop_path(self.attn(self.attn_norm(x), mask))
        if self.conv is not None:
            x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(0.5 * self.ff2(x))
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads, ff_mult=4, conv_kernel=31,
                 dropout=0.1, drop_path=0.0):
        super().__init__()
        self.pos_enc = RelativePositionalEncoding(dim)
        dp_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            ConformerBlock(dim, num_heads, ff_mult, conv_kernel, dropout, dp_rates[i])
            for i in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
