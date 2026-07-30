"""microK3: a small, readable Kimi K3-inspired language model.

This is an educational reimplementation, not an implementation of Kimi K3.
Run ``python microk3.py`` to train on the bundled tiny corpus.
"""

from __future__ import annotations

import argparse
import math
import sys
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


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. 12: bounded gate and value branches (|product| <= 100). The sigmoid reads the
        # uncapped pre-activation, so soft-capping never flattens the gate's own slope.
        pre = self.gate(x)
        g = 4 * torch.tanh(pre / 4) * torch.sigmoid(pre)
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

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError(f"tokens must have non-empty [batch, time] shape, got {tuple(tokens.shape)}")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"tokens must use an integer dtype, got {tokens.dtype}")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match tokens shape {tuple(tokens.shape)}"
            )
        if targets is not None and targets.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"targets must use an integer dtype, got {targets.dtype}")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        sources = [self.token(tokens)]
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
    def generate(self, tokens: torch.Tensor, count: int, temperature: float = 0.8) -> torch.Tensor:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError(f"tokens must have non-empty [batch, time] shape, got {tuple(tokens.shape)}")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"tokens must use an integer dtype, got {tokens.dtype}")
        low, high = torch.aminmax(tokens)
        minimum, maximum = low.item(), high.item()
        if minimum < 0 or maximum >= self.cfg.vocab_size:
            raise ValueError(
                f"tokens must be in [0, {self.cfg.vocab_size}), got range [{minimum}, {maximum}]"
            )
        if count == 0:
            return tokens
        # Prefill the prompt in one pass, then step token by token. Nothing here crops to
        # block_size: with no positional encoding anywhere, the model runs at any length.
        caches = self.caches()
        logits, _ = self(tokens, caches=caches)
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
    heads: int = model.cfg.heads
    matrices: dict[int, list[nn.Parameter]] = {}
    others: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
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
    if args.generate and not prompt:
        p.error("--prompt must encode to at least one byte")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = Config(block_size=args.block_size)
    model = MicroK3(cfg).to(args.device)
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
    model.train()
    for step in range(args.steps):
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
    if args.generate:
        seed = torch.tensor([list(prompt)], dtype=torch.long, device=args.device)
        sample = model.generate(seed, args.generate, args.temperature)[0].cpu().tolist()
        print("\n" + terminal_text(bytes(sample)))


if __name__ == "__main__":
    main()
