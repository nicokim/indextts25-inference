# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


class Attend(nn.Module):
    """Einsum attention with a padding mask (upstream's flash/causal variant was unused:
    the emotion Perceiver never enables it)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        scale = q.shape[-1] ** -0.5
        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        return einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return out * gamma + beta


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim))


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(dropout=dropout)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
