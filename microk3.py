"""microK3: a small, readable Kimi K3-inspired language model.

This is an educational reimplementation, not an implementation of Kimi K3.
Run ``python microk3.py`` to train on the bundled tiny corpus.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    latent_dim: int = 64
    expert_hidden: int = 192
    layers: int = 4
    heads: int = 4
    experts: int = 8
    top_k: int = 2
    block_size: int = 128
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * F.rms_norm(x, (x.size(-1),))


class KimiDeltaAttention(nn.Module):
    """Transparent recurrent delta rule with K3's lower-bounded decay.

    The report uses a fused chunkwise algorithm. This equivalent token loop is
    intentionally slow and legible: S <- decay*S + beta*(v - S^T k)*k.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.decay = nn.Linear(cfg.dim, cfg.heads, bias=True)
        self.beta = nn.Linear(cfg.dim, cfg.heads, bias=True)
        self.gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.normalize(q.view(b, t, self.heads, self.head_dim), dim=-1)
        k = F.normalize(k.view(b, t, self.heads, self.head_dim), dim=-1)
        v = v.view(b, t, self.heads, self.head_dim)
        # Eq. 5 of the report: log decay is bounded in (-5, 0).
        alpha = torch.exp(-5.0 * torch.sigmoid(self.decay(x))).unsqueeze(-1).unsqueeze(-1)
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1)
        state = x.new_zeros(b, self.heads, self.head_dim, self.head_dim)
        outputs = []
        for i in range(t):
            prediction = torch.einsum("bhij,bhi->bhj", state, k[:, i])
            error = v[:, i] - prediction
            update = torch.einsum("bhi,bhj->bhij", k[:, i], beta[:, i] * error)
            state = alpha[:, i] * state + update
            outputs.append(torch.einsum("bhij,bhi->bhj", state, q[:, i]))
        y = torch.stack(outputs, dim=1)
        y = self.norm(y).reshape(b, t, -1)
        return self.out(torch.sigmoid(self.gate(x)) * y)


class GatedAttention(nn.Module):
    """Tiny causal global attention standing in for K3's periodic Gated MLA."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.out(torch.sigmoid(self.gate(x)) * y)


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 12: bounded gate and value branches (|product| <= 100).
        g = 4 * torch.tanh(self.gate(x) / 4) * torch.sigmoid(self.gate(x))
        u = 25 * torch.tanh(self.up(x) / 25)
        return self.down(g * u)


class StableLatentMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.experts))
        self.down = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.experts = nn.ModuleList(SiTUGLU(cfg.latent_dim, cfg.expert_hidden) for _ in range(cfg.experts))
        self.up = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.latent_norm = RMSNorm(cfg.latent_dim)
        self.shared = SiTUGLU(cfg.dim, cfg.expert_hidden)
        self.last_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.router(x))
        _, chosen = torch.topk(scores + self.router_bias, self.top_k, dim=-1)
        mask = F.one_hot(chosen, len(self.experts)).sum(-2).to(scores.dtype)
        weights = scores * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        z = self.down(x)
        routed = sum(weights[..., i, None] * expert(z) for i, expert in enumerate(self.experts))
        self.last_load = mask.detach().sum((0, 1))
        return self.shared(x) + self.up(self.latent_norm(routed))


class Block(nn.Module):
    def __init__(self, cfg: Config, global_attention: bool):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.mix = GatedAttention(cfg) if global_attention else KimiDeltaAttention(cfg)
        self.moe = StableLatentMoE(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        return x + self.moe(self.norm2(x))


class MicroK3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.dim)
        # K3 uses 3 KDA : 1 global-attention layers. No positional embeddings.
        self.blocks = nn.ModuleList(Block(cfg, (i + 1) % 4 == 0) for i in range(cfg.layers))
        self.depth_queries = nn.Parameter(torch.zeros(cfg.layers, cfg.dim))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        sources = [self.token(tokens)]
        for i, block in enumerate(self.blocks):
            keys = torch.stack([F.rms_norm(s, (s.size(-1),)) for s in sources], dim=0)
            logits = torch.einsum("d,nbtd->nbt", self.depth_queries[i], keys) / math.sqrt(self.cfg.dim)
            weights = logits.softmax(0)
            x = sum(weights[j, ..., None] * s for j, s in enumerate(sources))
            sources.append(block(x))
        logits = self.lm_head(self.norm(sources[-1]))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, count: int, temperature: float = 0.8):
        for _ in range(count):
            logits, _ = self(tokens[:, -self.cfg.block_size :])
            probs = (logits[:, -1] / temperature).softmax(-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, 1)), dim=1)
        return tokens


def corpus(path: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    return ("microK3 learns by predicting the next byte.\n"
            "delta attention remembers; experts specialize; depth can attend.\n" * 80).encode()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", help="UTF-8 or arbitrary byte corpus")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Config()
    model = MicroK3(cfg).to(args.device)
    data = torch.tensor(list(corpus(args.data)), dtype=torch.long, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"\033[38;5;81m◆ microK3\033[0m  {sum(p.numel() for p in model.parameters()):,} parameters  {args.device}")
    model.train()
    for step in range(args.steps):
        starts = torch.randint(0, len(data) - cfg.block_size - 1, (8,), device=args.device)
        x = torch.stack([data[s:s + cfg.block_size] for s in starts])
        y = torch.stack([data[s + 1:s + cfg.block_size + 1] for s in starts])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"\033[38;5;213mstep {step:04d}\033[0m  loss {loss.item():.4f}")
    model.eval()
    sample = model.generate(torch.tensor([[ord('m')]], device=args.device), 120)[0].cpu().tolist()
    print("\n" + bytes(sample).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
