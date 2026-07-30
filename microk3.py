"""microK3: a small, readable Kimi K3-inspired language model.

This is an educational reimplementation, not an implementation of Kimi K3.
Run ``python microk3.py`` to train on the bundled tiny corpus.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
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
    dense_layers: int = 1
    block_size: int = 128
    dropout: float = 0.0
    # Vision tower. ``vision_layers=0`` leaves it out, which is the text-only default.
    vision_layers: int = 0
    vision_dim: int = 64
    vision_heads: int = 4
    vision_hidden: int = 128
    vision_channels: int = 1
    patch_size: int = 4
    # Deployment precision: microscaling quantization-aware training for routed experts.
    mx_qat: bool = False
    mx_block: int = 32

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "latent_dim": self.latent_dim,
            "expert_hidden": self.expert_hidden,
            "layers": self.layers,
            "heads": self.heads,
            "experts": self.experts,
            "block_size": self.block_size,
        }
        if self.vision_layers:
            positive |= {
                "vision_dim": self.vision_dim,
                "vision_heads": self.vision_heads,
                "vision_hidden": self.vision_hidden,
                "vision_channels": self.vision_channels,
                "patch_size": self.patch_size,
            }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim % self.heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")
        if not 1 <= self.top_k <= self.experts:
            raise ValueError(f"top_k must be between 1 and experts ({self.experts}), got {self.top_k}")
        if not 0 <= self.dense_layers <= self.layers:
            raise ValueError(f"dense_layers must be in [0, layers ({self.layers})], got {self.dense_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.vision_layers < 0:
            raise ValueError(f"vision_layers must be non-negative, got {self.vision_layers}")
        if self.vision_layers:
            if self.vision_dim % self.vision_heads != 0:
                raise ValueError(
                    f"vision_dim ({self.vision_dim}) must be divisible by vision_heads ({self.vision_heads})"
                )
            if (self.vision_dim // self.vision_heads) % 4 != 0:
                raise ValueError(
                    "2D RoPE splits each head into row and column rotation pairs, so "
                    f"vision_dim // vision_heads ({self.vision_dim // self.vision_heads}) must be a multiple of 4"
                )
        if self.mx_block < 2 or self.mx_block % 2 != 0:
            raise ValueError(f"mx_block must be an even element count, got {self.mx_block}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * F.rms_norm(x, (x.size(-1),))


@dataclass
class LayerCache:
    """One layer's decoding memory: a KDA recurrent state or a global-attention history.

    KDA's ``state`` stays the same size forever; only the global layers grow with the
    prompt. That asymmetry is the whole point of a 3:1 hybrid, so it is worth watching.
    """

    state: torch.Tensor | None = None
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None


def delta_rule_step(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one readable KDA recurrence step from report Eq. 1."""
    decayed = alpha * state
    prediction = torch.einsum("bhij,bhi->bhj", decayed, key)
    error = value - prediction
    update = torch.einsum("bhi,bhj->bhij", key, beta * error)
    state = decayed + update
    output = torch.einsum("bhij,bhi->bhj", state, query)
    return state, output


class KimiDeltaAttention(nn.Module):
    """Transparent recurrent form of K3's channel-wise delta rule.

    The report uses ShortConv + Swish projections and a fused chunkwise kernel.
    We keep plain projections and an intentionally slow, legible token loop.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.decay = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.log_decay_scale = nn.Parameter(torch.zeros(cfg.heads))
        self.beta = nn.Linear(cfg.dim, cfg.heads, bias=True)
        self.gate = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim)
        k = k.view(b, t, self.heads, self.head_dim)
        # Normalize in FP32 so a zero vector stays finite in FP16/BF16.
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype)
        v = v.view(b, t, self.heads, self.head_dim)
        # Eq. 5: per-key-channel log decay is bounded in (-5, 0).
        decay_logits = self.decay(x).view(b, t, self.heads, self.head_dim)
        decay_scale = self.log_decay_scale.exp().view(1, 1, self.heads, 1)
        alpha = torch.exp(-5.0 * torch.sigmoid(decay_scale * decay_logits)).unsqueeze(-1)
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1)
        state = None if cache is None else cache.state
        if state is None:
            state = x.new_zeros(b, self.heads, self.head_dim, self.head_dim)
        outputs = []
        for i in range(t):
            state, output = delta_rule_step(state, k[:, i], v[:, i], q[:, i], alpha[:, i], beta[:, i])
            outputs.append(output)
        if cache is not None:
            cache.state = state
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

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if cache is not None:
            k = k if cache.keys is None else torch.cat((cache.keys, k), dim=2)
            v = v if cache.values is None else torch.cat((cache.values, v), dim=2)
            cache.keys, cache.values = k, v
        # New queries sit at the end of the cached history, so the mask shifts with it.
        offset = k.size(2) - q.size(2)
        window = None if offset == 0 else q.new_ones(q.size(2), k.size(2), dtype=torch.bool).tril(offset)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=window, is_causal=window is None)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.out(torch.sigmoid(self.gate(x)) * y)


@dataclass(frozen=True)
class MXFormat:
    """One OCP microscaling element format (arXiv:2310.10537).

    A microscaling block is two things: tiny elements, and a single power-of-two scale that
    a run of them share along the reduction axis. ``mantissa_bits`` and ``min_exponent``
    are all it takes to describe the element grid.
    """

    name: str
    bits: int
    mantissa_bits: int
    min_exponent: int
    largest: float

    @property
    def emax(self) -> int:
        """Exponent of the largest magnitude: 6 = 1.5 * 2**2, and 448 = 1.75 * 2**8."""
        return int(math.log2(self.largest))


MXFP4 = MXFormat("MXFP4", bits=4, mantissa_bits=1, min_exponent=0, largest=6.0)  # E2M1 elements
MXFP8 = MXFormat("MXFP8", bits=8, mantissa_bits=3, min_exponent=-6, largest=448.0)  # E4M3 elements
# The eight E2M1 magnitudes in code order: the index is literally the element's low three bits.
MXFP4_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mx_split(x: torch.Tensor, block: int) -> torch.Tensor:
    """Group the reduction axis into blocks, zero-padding a ragged tail.

    Real MX layouts require that axis to be a multiple of the block size; padding is a
    teaching-scale convenience so tiny widths still quantize.
    """
    return F.pad(x, (0, -x.size(-1) % block)).unflatten(-1, (-1, block))


def mx_shared_scale(blocks: torch.Tensor, fmt: MXFormat) -> torch.Tensor:
    """One E8M0 scale per block: the power of two that lifts the block maximum to ``emax``."""
    largest = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.floor(torch.log2(largest.clamp_min(torch.finfo(blocks.dtype).smallest_normal)))
    # E8M0 carries an exponent and nothing else—no mantissa, no sign, range -127..127.
    return torch.exp2((exponent - fmt.emax).clamp(-127, 127))


def quantize_elements(x: torch.Tensor, fmt: MXFormat) -> torch.Tensor:
    """Round already-scaled elements onto ``fmt``'s grid, then clamp instead of overflowing."""
    exponent = torch.floor(torch.log2(x.abs().clamp_min(torch.finfo(x.dtype).smallest_normal)))
    step = torch.exp2(exponent.clamp_min(fmt.min_exponent) - fmt.mantissa_bits)
    # torch.round is round-half-to-even, which is the tie rule the format asks for. The clamp
    # matters: a block maximum in [448, 512) scales past E4M3's reach and saturates there.
    return (x / step).round().mul(step).clamp(-fmt.largest, fmt.largest)


def quantize_mx(x: torch.Tensor, fmt: MXFormat, block: int) -> torch.Tensor:
    """Round ``x`` onto a microscaling grid along its last axis. Deliberately gradient-free."""
    blocks = mx_split(x.detach().float(), block)
    scale = mx_shared_scale(blocks, fmt)
    quantized = quantize_elements(blocks / scale, fmt) * scale
    return quantized.flatten(-2)[..., : x.size(-1)].to(x.dtype)


def fake_quantize_mx(x: torch.Tensor, fmt: MXFormat, block: int) -> torch.Tensor:
    """QAT: the forward pass sees the quantized grid, the backward pass sees an identity."""
    return x + (quantize_mx(x, fmt, block) - x).detach()


def mx_linear(layer: nn.Linear, x: torch.Tensor, block: int) -> torch.Tensor:
    """One expert matmul in K3's deployment precision: MXFP4 weights, MXFP8 input activations."""
    if not block:
        return layer(x)
    return F.linear(fake_quantize_mx(x, MXFP8, block), fake_quantize_mx(layer.weight, MXFP4, block))


def pack_mxfp4(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Store ``x`` for real: two E2M1 codes per byte, plus one E8M0 exponent byte per block."""
    blocks = mx_split(x.detach().float(), block)
    scale = mx_shared_scale(blocks, MXFP4)
    elements = quantize_elements(blocks / scale, MXFP4)
    magnitudes = torch.tensor(MXFP4_CODES, device=x.device)
    codes = torch.searchsorted(magnitudes, elements.abs().contiguous())
    codes = codes | (elements < 0).long() << 3  # the sign bit sits above the three magnitude bits
    packed = (codes[..., 0::2] | codes[..., 1::2] << 4).to(torch.uint8)
    return packed, (torch.log2(scale).squeeze(-1) + 127).to(torch.uint8)


def unpack_mxfp4(packed: torch.Tensor, exponents: torch.Tensor, width: int) -> torch.Tensor:
    """Rebuild a dequantized tensor of row width ``width`` from nibbles and exponent bytes."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor(MXFP4_CODES, device=packed.device)[codes & 0x7]
    scale = torch.exp2(exponents.float() - 127).unsqueeze(-1)
    values = torch.where(codes & 0x8 == 0, magnitudes, -magnitudes) * scale
    return values.flatten(-2)[..., :width]


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, mx_block: int = 0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        # Non-zero only for routed experts: K3 ships those weights as MXFP4 and nothing else.
        self.mx_block = mx_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 12: bounded gate and value branches (|product| <= 100). The sigmoid reads the
        # uncapped pre-activation, so soft-capping never flattens the gate's own slope.
        pre = mx_linear(self.gate, x, self.mx_block)
        g = 4 * torch.tanh(pre / 4) * torch.sigmoid(pre)
        u = 25 * torch.tanh(mx_linear(self.up, x, self.mx_block) / 25)
        return mx_linear(self.down, g * u, self.mx_block)


def mxfp4_footprint(model: nn.Module) -> tuple[int, int]:
    """Weight count and packed byte count for everything this model would ship as MXFP4."""
    parameters = stored = 0
    for module in model.modules():
        if not isinstance(module, SiTUGLU) or not module.mx_block:
            continue
        for layer in (module.gate, module.up, module.down):
            codes, exponents = pack_mxfp4(layer.weight, module.mx_block)
            parameters += layer.weight.numel()
            stored += codes.numel() + exponents.numel()
    return parameters, stored


class StableLatentMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.experts))
        self.down = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        # Only the routed experts quantize. The report keeps the router, the latent
        # projections, the shared expert, and every attention matrix in higher precision.
        expert_block = cfg.mx_block if cfg.mx_qat else 0
        self.experts = nn.ModuleList(
            SiTUGLU(cfg.latent_dim, cfg.expert_hidden, expert_block) for _ in range(cfg.experts)
        )
        self.up = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.latent_norm = RMSNorm(cfg.latent_dim)
        self.shared = SiTUGLU(cfg.dim, cfg.expert_hidden)
        self.last_load: torch.Tensor | None = None

    @torch.no_grad()
    def _update_router_bias(self, scores: torch.Tensor, cutoff: torch.Tensor) -> None:
        """Apply exact, batch-local Quantile Balancing for the next forward pass."""
        margins = scores.reshape(-1, len(self.experts)).float() - cutoff.reshape(-1, 1).float()
        quantile = torch.quantile(margins, 1.0 - self.top_k / len(self.experts), dim=0)
        next_bias = -quantile
        next_bias -= next_bias.mean()
        self.router_bias.copy_(next_bias.to(self.router_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.router(x))
        biased_scores = scores + self.router_bias
        if self.top_k < len(self.experts):
            ranked_scores, ranked = torch.topk(biased_scores, self.top_k + 1, dim=-1)
            chosen = ranked[..., : self.top_k]
            cutoff = ranked_scores[..., self.top_k]
        else:
            _, chosen = torch.topk(biased_scores, self.top_k, dim=-1)
            cutoff = None
        mask = F.one_hot(chosen, len(self.experts)).sum(-2).to(scores.dtype)
        weights = scores * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        z = self.down(x)
        # K3 dispatches tokens to their Top-k experts; we run every expert densely and mask
        # afterwards, which is the same function and much easier to read (and much slower).
        routed = sum(weights[..., i, None] * expert(z) for i, expert in enumerate(self.experts))
        self.last_load = mask.detach().sum((0, 1))
        if self.training and cutoff is not None:
            self._update_router_bias(scores.detach(), cutoff.detach())
        return self.shared(x) + self.up(self.latent_norm(routed))


def rope_2d(
    rows: torch.Tensor, columns: torch.Tensor, head_dim: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-dimensional RoPE: half of each head's rotation pairs follow the patch row, half the column."""
    pairs = head_dim // 4  # rotation pairs per axis; the two axes fill head_dim // 2 pairs
    frequency = base ** (-torch.arange(pairs, device=rows.device, dtype=torch.float32) / pairs)
    angles = torch.cat((rows[:, None] * frequency, columns[:, None] * frequency), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs, so an attention logit depends on the patch offset."""
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class VisionBlock(nn.Module):
    """One patch-transformer layer: bidirectional attention with 2D RoPE, then a bounded GLU.

    MoonViT-V2 uses RMSNorm and drops every bias term, which is what stabilizes training a
    vision tower from scratch; both choices carry over here unchanged.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.head_dim = cfg.vision_heads, cfg.vision_dim // cfg.vision_heads
        self.norm1, self.norm2 = RMSNorm(cfg.vision_dim), RMSNorm(cfg.vision_dim)
        self.qkv = nn.Linear(cfg.vision_dim, 3 * cfg.vision_dim, bias=False)
        self.out = nn.Linear(cfg.vision_dim, cfg.vision_dim, bias=False)
        self.ffn = SiTUGLU(cfg.vision_dim, cfg.vision_hidden)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        # Patches see each other in both directions; only the language model is causal.
        y = F.scaled_dot_product_attention(
            apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, attn_mask=mask
        )
        x = x + self.out(y.transpose(1, 2).reshape(b, n, -1))
        return x + self.ffn(self.norm2(x))


class MicroMoonViT(nn.Module):
    """Native-resolution patch tower, trained from scratch by next-byte prediction.

    Each image keeps its own patch grid, and a batch of differently sized images travels as
    one packed sequence whose block-diagonal mask holds attention inside a single image—the
    report's intra-frame spatial pass, with one frame per sample. K3 then pixel-shuffles 2x2
    patch groups before an MLP projector; microK3 shuffles the same way into one bias-free
    matrix. Video, temporal attention, and temporal pooling are absent.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.vision_dim // cfg.vision_heads
        self.embed = nn.Linear(cfg.vision_channels * cfg.patch_size**2, cfg.vision_dim, bias=False)
        self.blocks = nn.ModuleList(VisionBlock(cfg) for _ in range(cfg.vision_layers))
        self.norm = RMSNorm(cfg.vision_dim)
        self.project = nn.Linear(4 * cfg.vision_dim, cfg.dim, bias=False)

    def patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Flatten one image into embedded patch tokens, keeping its own grid shape."""
        if image.ndim != 3:
            raise ValueError(
                f"each image must have [channels, height, width] shape, got {tuple(image.shape)}"
            )
        if not image.is_floating_point():
            raise ValueError(f"images must use a floating dtype, got {image.dtype}")
        channels, height, width = image.shape
        patch, group = self.cfg.patch_size, 2 * self.cfg.patch_size
        if channels != self.cfg.vision_channels:
            raise ValueError(f"expected {self.cfg.vision_channels} image channels, got {channels}")
        if height % group or width % group:
            raise ValueError(
                f"image sides must be multiples of 2 * patch_size ({group}) for the 2x2 pixel "
                f"shuffle, got {height}x{width}"
            )
        rows, columns = height // patch, width // patch
        grid = image.unflatten(1, (rows, patch)).unflatten(-1, (columns, patch))
        patches = grid.permute(1, 3, 0, 2, 4).flatten(2)  # [rows, columns, channels * patch * patch]
        return self.embed(patches).flatten(0, 1), rows, columns

    def shuffle(self, tokens: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
        """Pixel-shuffle 2x2 patch groups into the channel axis, then project to the byte width."""
        grid = tokens.unflatten(0, (rows // 2, 2, columns // 2, 2))
        merged = grid.permute(0, 2, 1, 3, 4).flatten(2)  # [rows / 2, columns / 2, 4 * vision_dim]
        return self.project(merged.flatten(0, 1))

    def forward(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        tokens, all_rows, all_columns, grids = [], [], [], []
        for image in images:
            patches, rows, columns = self.patchify(image)
            row, column = torch.meshgrid(
                torch.arange(rows, device=image.device, dtype=torch.float32),
                torch.arange(columns, device=image.device, dtype=torch.float32),
                indexing="ij",
            )
            tokens.append(patches)
            all_rows.append(row.flatten())
            all_columns.append(column.flatten())
            grids.append((rows, columns))
        cos, sin = rope_2d(torch.cat(all_rows), torch.cat(all_columns), self.head_dim)
        counts = [rows * columns for rows, columns in grids]
        owner = torch.cat([torch.full((n,), i, device=cos.device) for i, n in enumerate(counts)])
        packed = torch.cat(tokens)[None]
        mask = (owner[:, None] == owner[None, :])[None, None]
        for block in self.blocks:
            packed = block(packed, cos, sin, mask)
        encoded = self.norm(packed)[0].split(counts)
        return [self.shuffle(image_tokens, *grid) for image_tokens, grid in zip(encoded, grids, strict=True)]


class Block(nn.Module):
    def __init__(self, cfg: Config, global_attention: bool, dense: bool):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.mix = GatedAttention(cfg) if global_attention else KimiDeltaAttention(cfg)
        # K3 sets first_k_dense_replace=1: the first layer is a plain FFN, so the router
        # never has to learn dispatch from a barely-shaped embedding stream.
        self.ffn = SiTUGLU(cfg.dim, cfg.expert_hidden) if dense else StableLatentMoE(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        """Return this layer's contribution, not a cumulative residual state."""
        mixed = self.dropout(self.mix(self.norm1(x), cache))
        routed = self.dropout(self.ffn(self.norm2(x + mixed)))
        return mixed + routed


class MicroK3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.dim)
        # Native vision: one shared backbone, so patch tokens join the byte stream as a
        # prefix rather than passing through a second model or an alignment stage.
        self.vision = MicroMoonViT(cfg) if cfg.vision_layers else None
        # K3 repeats 3 KDA : 1 global attention and always ends globally.
        self.blocks = nn.ModuleList(
            Block(cfg, (i + 1) % 4 == 0 or i == cfg.layers - 1, i < cfg.dense_layers)
            for i in range(cfg.layers)
        )
        self.depth_queries = nn.Parameter(torch.zeros(cfg.layers + 1, cfg.dim))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Keep tied byte-token logits near the uniform-loss baseline at startup."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def caches(self) -> list[LayerCache]:
        """Fresh decoding memory, one slot per layer."""
        return [LayerCache() for _ in self.blocks]

    def encode_images(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Encode images into byte-stream width: one variable-length token sequence each."""
        if self.vision is None:
            raise ValueError("this model has no vision tower; build one with Config(vision_layers=...)")
        return self.vision(images)

    def _context_length(self, tokens: torch.Tensor, prefix: torch.Tensor | None) -> int:
        """Validate a text/image context and return its combined sequence length."""
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError(
                f"tokens must have [batch, time] shape with a non-empty batch, got {tuple(tokens.shape)}"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"tokens must use an integer dtype, got {tokens.dtype}")
        if prefix is not None and (
            prefix.ndim != 3 or prefix.shape[0] != tokens.shape[0] or prefix.shape[2] != self.cfg.dim
        ):
            raise ValueError(
                f"prefix must have shape [batch ({tokens.shape[0]}), prefix, dim ({self.cfg.dim})], "
                f"got {tuple(prefix.shape)}"
            )
        length = tokens.shape[1] + (0 if prefix is None else prefix.shape[1])
        if length == 0:
            raise ValueError("tokens or prefix must provide a non-empty sequence")
        return length

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[LayerCache] | None = None,
        prefix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # An image prefix is context, not text, so an empty token block is meaningful there.
        length = self._context_length(tokens, prefix)
        if targets is not None and targets.shape != (tokens.shape[0], length):
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match [batch, prefix + time] "
                f"{(tokens.shape[0], length)}"
            )
        if targets is not None and targets.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"targets must use an integer dtype, got {targets.dtype}")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        embedded = self.token(tokens)
        sources = [embedded if prefix is None else torch.cat((prefix, embedded), dim=1)]
        for i, block in enumerate(self.blocks):
            x = self._mix_depth(self.depth_queries[i], sources)
            sources.append(block(x, None if caches is None else caches[i]))
        final = self._mix_depth(self.depth_queries[-1], sources)
        logits = self.lm_head(self.norm(final))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @staticmethod
    def _mix_depth(query: torch.Tensor, sources: list[torch.Tensor]) -> torch.Tensor:
        """Full AttnRes-style retrieval over the embedding and layer contributions."""
        keys = torch.stack([F.rms_norm(source, (source.size(-1),)) for source in sources], dim=0)
        weights = torch.einsum("d,nbtd->nbt", query, keys).softmax(0)
        return sum(weights[j, ..., None] * source for j, source in enumerate(sources))

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        count: int,
        temperature: float = 0.8,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
        self._context_length(tokens, prefix)
        if tokens.numel():
            low, high = torch.aminmax(tokens)
            minimum, maximum = low.item(), high.item()
            if minimum < 0 or maximum >= self.cfg.vocab_size:
                raise ValueError(
                    f"tokens must be in [0, {self.cfg.vocab_size}), got range [{minimum}, {maximum}]"
                )
        if count == 0:
            return tokens
        # Prefill the prompt in one pass, then step token by token. Nothing here crops to
        # block_size: with no positional encoding anywhere, the model runs at any length. An
        # image prefix is encoded once, into the prefill; the caches carry it after that.
        caches = self.caches()
        logits, _ = self(tokens, caches=caches, prefix=prefix)
        for _ in range(count):
            probs = (logits[:, -1] / temperature).softmax(-1)
            latest = torch.multinomial(probs, 1)
            tokens = torch.cat((tokens, latest), dim=1)
            logits, _ = self(latest, caches=caches)
        return tokens


def orthogonalize(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration that pushes every singular value toward one."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if tall else x


class Muon(torch.optim.Optimizer):
    """Per-head Muon: momentum, then an orthogonalized step for each head's own slice.

    K3 trains matrices with Per-Head Muon and everything else with AdamW. ``blocks`` is
    how many head-sized row groups a weight holds—fused Q/K/V holds three per head—so the
    Newton-Schulz iteration runs batched over heads rather than over the whole matrix.
    QK-clip, the other half of K3's optimizer, is not reproduced here.
    """

    def __init__(self, groups, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, blocks=1))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                state["momentum"].lerp_(parameter.grad, 1 - group["momentum"])
                # Nesterov-style lookahead, as in the reference Muon.
                update = parameter.grad.lerp(state["momentum"], group["momentum"])
                update = orthogonalize(update.view(group["blocks"], -1, parameter.size(-1)))
                scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"] * scale)


def parameter_groups(model: MicroK3) -> tuple[dict[int, list[nn.Parameter]], list[nn.Parameter]]:
    """Sort weights into head-aligned matrix groups and the scalar-ish remainder."""
    matrices: dict[int, list[nn.Parameter]] = {}
    others: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        # The tower has its own head count, so its packed Q/K/V splits into its own blocks.
        heads = model.cfg.vision_heads if name.startswith("vision.") else model.cfg.heads
        if parameter.ndim < 2 or name in ("token.weight", "depth_queries"):
            others.append(parameter)
        elif name.endswith("qkv.weight"):
            matrices.setdefault(3 * heads, []).append(parameter)
        elif ".mix." in name and name.endswith(("gate.weight", "decay.weight")):
            matrices.setdefault(heads, []).append(parameter)
        else:
            matrices.setdefault(1, []).append(parameter)
    return matrices, others


def build_optimizers(
    model: MicroK3,
    kind: str,
    learning_rate: float,
    muon_lr: float = 0.02,
    weight_decay: float = 0.1,
) -> list[torch.optim.Optimizer]:
    """Matrices get Muon or AdamW; embeddings, gains and biases always get undecayed AdamW."""
    matrices, others = parameter_groups(model)
    tail = torch.optim.AdamW(others, lr=learning_rate, weight_decay=0.0)
    if kind == "muon":
        groups = [{"params": params, "blocks": blocks} for blocks, params in sorted(matrices.items())]
        return [Muon(groups, lr=muon_lr, weight_decay=weight_decay), tail]
    flat = [parameter for params in matrices.values() for parameter in params]
    return [torch.optim.AdamW(flat, lr=learning_rate, weight_decay=weight_decay), tail]


def learning_rate_scale(step: int, total: int, warmup: float = 0.01) -> float:
    """K3's schedule shape: a 1% linear warmup, then cosine decay."""
    warmup_steps = max(1, round(total * warmup))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total - warmup_steps)))


def corpus(path: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    return (
        "microK3 learns by predicting the next byte.\n"
        "delta attention remembers; experts specialize; depth can attend.\n" * 80
    ).encode()


def sample_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-byte windows, including the final valid window."""
    if data.ndim != 1:
        raise ValueError(f"data must be a one-dimensional token stream, got shape {tuple(data.shape)}")
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("block_size and batch_size must be positive")
    if data.numel() <= block_size:
        raise ValueError(
            f"corpus has {data.numel()} bytes; it needs at least {block_size + 1} for block_size={block_size}"
        )
    starts = torch.randint(0, data.numel() - block_size, (batch_size,), device=data.device)
    offsets = starts[:, None] + torch.arange(block_size, device=data.device)
    return data[offsets], data[offsets + 1]


SHAPES = ("square", "circle", "triangle", "cross")
# Sides are multiples of 2 * patch_size, and the tower sees a different resolution each step.
SHAPE_SIZES = (24, 32, 40)


def draw_shape(kind: int, size: int) -> torch.Tensor:
    """One noisy grayscale [1, size, size] picture of a filled shape, placed and scaled at random."""
    axis = torch.linspace(-1.0, 1.0, size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    radius = 0.35 + 0.3 * torch.rand(())
    centre_y, centre_x = ((1 - radius) * (2 * torch.rand(2) - 1)).tolist()
    dy, dx = (y - centre_y).abs(), (x - centre_x).abs()
    if SHAPES[kind] == "square":
        filled = (dy <= radius) & (dx <= radius)
    elif SHAPES[kind] == "circle":
        filled = dy**2 + dx**2 <= radius**2
    elif SHAPES[kind] == "triangle":
        filled = (y - centre_y <= radius) & (y - centre_y >= 2 * dx - radius)
    else:
        filled = ((dx <= radius / 3) & (dy <= radius)) | ((dy <= radius / 3) & (dx <= radius))
    return (filled.float() + 0.05 * torch.randn(size, size)).clamp(0, 1)[None]


def shapes_batch(
    batch_size: int, size: int, device: str | torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch of shape pictures with their byte captions and supervised caption lengths.

    One resolution per batch keeps the projected prefix rectangular. The tower itself packs
    mixed resolutions in a single pass; only this rectangular hand-off asks them to agree.
    """
    if batch_size <= 0 or size <= 0:
        raise ValueError("batch_size and size must be positive")
    kinds = torch.randint(0, len(SHAPES), (batch_size,)).tolist()
    images = torch.stack([draw_shape(kind, size) for kind in kinds])
    captions = [SHAPES[kind].encode() + b"\n" for kind in kinds]
    tokens = torch.full((batch_size, max(len(name) for name in SHAPES) + 1), ord("\n"))
    for row, text in enumerate(captions):
        tokens[row, : len(text)] = torch.tensor(list(text))
    lengths = torch.tensor([len(text) for text in captions])
    return images.to(device), tokens.to(device), lengths.to(device)


def prefix_targets(tokens: torch.Tensor, lengths: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Line targets up so the last image token predicts the caption's first byte.

    Positions with nothing to predict—every image token but the last, and padding past each
    caption—carry -100, which cross entropy ignores.
    """
    if prefix_len < 1:
        raise ValueError(f"prefix_len must be at least 1, got {prefix_len}")
    batch, width = tokens.shape
    supervised = tokens.masked_fill(torch.arange(width, device=tokens.device) >= lengths[:, None], -100)
    targets = tokens.new_full((batch, prefix_len + width), -100)
    targets[:, prefix_len - 1 : prefix_len + width - 1] = supervised
    return targets


def caption(model: MicroK3, images: torch.Tensor, count: int, temperature: float) -> list[str]:
    """Read each picture with nothing else in context: the image is the entire prompt."""
    prefix = torch.stack(model.encode_images(images))
    empty = torch.zeros(len(images), 0, dtype=torch.long, device=prefix.device)
    sampled = model.generate(empty, count, temperature, prefix=prefix)
    return [bytes(row).split(b"\n")[0].decode("utf-8", errors="replace") for row in sampled.cpu().tolist()]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_device(value: str) -> str:
    try:
        device = torch.device(value)
    except RuntimeError as error:  # argparse renders this as a one-line usage error
        raise argparse.ArgumentTypeError(str(error)) from None
    if device.type not in {"cpu", "cuda", "mps"}:
        raise argparse.ArgumentTypeError("device must be cpu, cuda, or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA was requested, but no CUDA device is available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise argparse.ArgumentTypeError("MPS was requested, but no MPS device is available")
    return value


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def styled(text: str, color: int) -> str:
    return f"\033[38;5;{color}m{text}\033[0m" if sys.stdout.isatty() else text


def terminal_text(data: bytes) -> str:
    """Decode arbitrary bytes while escaping terminal control characters."""
    decoded = data.decode("utf-8", errors="replace")
    return "".join(
        character
        if character in "\n\t" or character.isprintable()
        else character.encode("unicode_escape").decode("ascii")
        for character in decoded
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", help="UTF-8 or arbitrary byte corpus")
    p.add_argument("--steps", type=positive_int, default=100)
    p.add_argument("--batch-size", type=positive_int, default=8)
    p.add_argument("--block-size", type=positive_int, default=128)
    p.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-3,
        help="AdamW step size; with --optimizer muon it drives only the embedding and gain tail",
    )
    p.add_argument(
        "--optimizer",
        choices=("adamw", "muon"),
        default="muon",
        help="muon runs K3's per-head orthogonalized step on matrices",
    )
    p.add_argument(
        "--muon-lr", type=positive_float, default=0.02, help="matrix step size for --optimizer muon"
    )
    p.add_argument(
        "--vision",
        action="store_true",
        help="train the patch tower to caption procedural shapes instead of reading a corpus; "
        "--data and --block-size are then unused",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="quantization-aware training: MXFP4 routed-expert weights, MXFP8 input activations",
    )
    p.add_argument(
        "--generate",
        type=non_negative_int,
        default=120,
        help="bytes to sample after training; use 0 to skip",
    )
    p.add_argument("--temperature", type=positive_float, default=0.8)
    p.add_argument("--prompt", default="m", help="UTF-8 generation prompt")
    p.add_argument("--device", type=available_device, default=default_device())
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prompt = args.prompt.encode()
    if args.generate and not args.vision and not prompt:
        p.error("--prompt must encode to at least one byte")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = Config(block_size=args.block_size, vision_layers=2 if args.vision else 0, mx_qat=args.quantize)
    model = MicroK3(cfg).to(args.device)
    data = None
    if not args.vision:
        data = torch.tensor(list(corpus(args.data)), dtype=torch.long, device=args.device)
        if data.numel() <= cfg.block_size:
            p.error(
                f"corpus has {data.numel()} bytes; use at least {cfg.block_size + 1} bytes "
                "or choose a smaller --block-size"
            )
    optimizers = build_optimizers(model, args.optimizer, args.learning_rate, args.muon_lr)
    schedules = [
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_scale(step, args.steps))
        for optimizer in optimizers
    ]
    count = sum(parameter.numel() for parameter in model.parameters())
    python_version = ".".join(map(str, sys.version_info[:3]))
    print(
        f"{styled('◆ microK3', 81)}  {count:,} parameters  {args.device}  "
        f"python {python_version}  torch {torch.__version__}  seed {args.seed}  "
        f"block {cfg.block_size}  batch {args.batch_size}  {args.optimizer}"
    )
    if cfg.vision_layers:
        tokens_per_image = "/".join(str((size // cfg.patch_size // 2) ** 2) for size in SHAPE_SIZES)
        print(
            f"{styled('◇ vision', 81)}  {cfg.vision_layers}-layer tower  patch {cfg.patch_size}  "
            f"{'/'.join(map(str, SHAPE_SIZES))} px  {tokens_per_image} tokens per image after pixel shuffle"
        )
    if cfg.mx_qat:
        quantized, stored = mxfp4_footprint(model)
        bits = MXFP4.bits + 8 / cfg.mx_block
        print(
            f"{styled('◇ MX-QAT', 81)}  {MXFP4.name} weights, {MXFP8.name} activations  "
            f"block {cfg.mx_block}  {bits:.2f} bits per routed weight  {quantized:,} weights  "
            f"{stored:,} B packed against {4 * quantized:,} B in float32"
        )
    model.train()
    for step in range(args.steps):
        if cfg.vision_layers:
            size = SHAPE_SIZES[int(torch.randint(len(SHAPE_SIZES), ()))]
            images, x, lengths = shapes_batch(args.batch_size, size, args.device)
            prefix = torch.stack(model.encode_images(images))
            _, loss = model(x, prefix_targets(x, lengths, prefix.shape[1]), prefix=prefix)
        else:
            assert data is not None
            x, y = sample_batch(data, cfg.block_size, args.batch_size)
            _, loss = model(x, y)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer, schedule in zip(optimizers, schedules, strict=True):
            optimizer.step()
            schedule.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"{styled(f'step {step:04d}', 213)}  loss {loss.item():.4f}")
    model.eval()
    if args.generate and cfg.vision_layers:
        images, x, lengths = shapes_batch(8, SHAPE_SIZES[-1], args.device)
        wanted = [
            bytes(row[:n]).decode().strip() for row, n in zip(x.tolist(), lengths.tolist(), strict=True)
        ]
        read = caption(model, images, max(map(len, SHAPES)) + 1, args.temperature)
        correct = sum(want == got for want, got in zip(wanted, read, strict=True))
        print(f"\n{styled(f'{correct}/{len(wanted)} pictures read', 213)}  at {SHAPE_SIZES[-1]} px")
        for want, got in zip(wanted, read, strict=True):
            print(f"  {want:<8} → {got!r}")
    elif args.generate:
        seed = torch.tensor([list(prompt)], dtype=torch.long, device=args.device)
        sample = model.generate(seed, args.generate, args.temperature)[0].cpu().tolist()
        print("\n" + terminal_text(bytes(sample)))


if __name__ == "__main__":
    main()
