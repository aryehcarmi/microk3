<div align="center">
  <img src="assets/hero.svg" width="100%" alt="microK3 — frontier ideas, laptop scale">
  <p><strong>A tiny, readable model for learning four ideas behind Kimi K3.</strong></p>
  <p>
    <a href="#five-minute-tour">Five-minute tour</a> ·
    <a href="#what-is-faithful">Honesty map</a> ·
    <a href="#pictures-and-four-bit-experts">Pictures and four-bit experts</a> ·
    <a href="#report-to-code-map">Report → code</a> ·
    <a href="#modal">Modal</a> ·
    <a href="docs/report-notes.md">Source notes</a>
  </p>
  <p>
    <a href="https://github.com/aryehcarmi/microk3/actions/workflows/ci.yml">
      <img src="https://github.com/aryehcarmi/microk3/actions/workflows/ci.yml/badge.svg" alt="CI status">
    </a>
  </p>
</div>

> [!IMPORTANT]
> **microK3 is K3-inspired, not Kimi K3.** It is a ~1.65M-parameter teaching model, not a
> reproduction or distillation. It never downloads the 2.8T-parameter weights. The goal is the
> Karpathy-style feeling of seeing the whole learning system—not benchmark parity.

## Why this exists

Kimi K3 combines four ideas: recurrent **Kimi Delta Attention**, periodic global
attention, **Attention Residuals** across depth, and a **Stable LatentMoE** across width. The
official implementation needs industrial infrastructure. This repository turns the conceptual
spine into one hackable file, one diagram, and a small regression suite. Two further pieces of the
release—the MoonViT-V2 vision tower and MXFP4 deployment precision—are here too, at the same
teaching scale and behind their own flags.

![microK3 on one page: the forward pass over a depth bus, the KDA recurrence, the depth-attention
pattern, Stable LatentMoE routing, decoding memory, the optimizer split, and the scale gap to Kimi
K3](assets/architecture.svg)

## Five-minute tour

```bash
git clone https://github.com/aryehcarmi/microk3
cd microk3
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
microk3 --steps 20
```

On CPU-only Linux, replace the install line with the following so pip does not pull the much
larger CUDA build of PyTorch:

```bash
pip install 'torch>=2.4.1,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
```

The test run should report `65 passed`. The first training loss should be near the uniform
byte-token baseline `ln(256) ≈ 5.55`; an initial loss in the tens or hundreds is a bug, not a
learning challenge. Twenty steps is enough to watch that number fall and nothing more—the sample
printed at the end is still byte noise, and stays noise until a few hundred steps on a real corpus.

On an Apple M2 laptop with torch 2.10, that quickstart takes about 9 seconds on CPU: roughly 2.5
training steps per second, plus a 120-byte sample. `--device mps` trains a shade faster and samples
about three times slower, because the KDA token loop launches many small kernels and pays a
one-time shader compilation on its first run. At this scale either device is fine.

Bring a text, code, or arbitrary byte corpus:

```bash
microk3 --data path/to/tiny.txt --steps 1000
```

The file must contain at least `block_size + 1` bytes—129 bytes at the default setting, though a
larger corpus is much more useful. Use `--block-size 64` for a faster CPU experiment. Training
prints a sample at the end; it intentionally does not create a checkpoint or background job.
Use `--generate 0` for a train-only run. Training defaults to the per-head Muon step with a warmup
then cosine schedule; `--optimizer adamw` switches back. `--vision` and `--quantize` turn on the
two optional parts described in [Pictures and four-bit experts](#pictures-and-four-bit-experts).
Run `microk3 --help` to see the batch size, learning rate, optimizer, prompt, temperature, and
device controls.

Read [`microk3.py`](microk3.py) top to bottom. The suggested path is:

1. `KimiDeltaAttention`: Eq. 1 as a legible recurrent token loop, including channel-wise decay
   before the delta correction; Eq. 5 supplies the bounded log-decay.
2. `GatedAttention`: periodic causal global attention. K3 uses compressed MLA; microK3 uses
   ordinary attention so the lesson fits on one screen.
3. `SiTUGLU` → `StableLatentMoE`: route through a small latent, normalize the routed aggregate,
   and update dispatch biases with an exact batch-local version of Quantile Balancing.
4. `MicroK3._mix_depth`: full AttnRes-style retrieval over the embedding and individual layer
   contributions, with a separate final-output query.
5. `LayerCache` → `MicroK3.generate`: prefill once, then decode step by step. Every KDA state
   stays the same size forever; only the global layers' caches grow with the sequence.
6. `orthogonalize` → `Muon` → `build_optimizers`: K3's per-head orthogonalized step for matrix
   weights, with an undecayed AdamW tail for embeddings, gains, and biases.
7. `MXFormat` → `quantize_mx` → `mx_linear` → `pack_mxfp4`: the deployment precision, from the
   element grid and its shared block scale to the bit layout the weights would actually ship in.
8. `rope_2d` → `MicroMoonViT` → `prefix_targets`: patches at the image's own resolution, several
   pictures packed into one attention pass, then projected into the byte stream as a prefix.

Model shape—`layers`, `dim`, `experts`, `top_k`, `dense_layers`, and the `vision_*` widths—is
deliberately not exposed as CLI flags. Edit the `Config` defaults at the top of `microk3.py` so
every change stays visible in the file you are reading. `--vision` and `--quantize` decide only
whether a part is built and used at all; the tower's shape and the microscaling block size stay in
`Config`.

## What is faithful?

| Idea | Kimi K3 | microK3 | Status |
|---|---|---|---|
| Hybrid schedule | 69 KDA + 24 Gated MLA; 3:1 blocks plus final global layer | 3 KDA + 1 global by default; arbitrary depths still end globally | 🟢 pattern |
| Delta recurrence | Channel-wise decay, then delta correction | Same recurrence in a readable token loop | 🟢 equation |
| KDA projections | ShortConv + Swish Q/K/V; low-rank decay logits | Plain linear Q/K/V; full-rank decay logits | 🟡 simplified |
| Lower-bounded decay | `g = -5 sigmoid(exp(A) z)` per key channel | Same mapping, per key channel | 🟢 equation |
| KDA execution | Fused chunkwise kernel | Sequential Python loop | 🟡 equivalent recurrence, slow execution |
| Global attention | Gated MLA, NoPE | Gated MHA, no positional embeddings | 🟡 substituted |
| Attention Residuals | Eight block-level groups with partial sums | Full attention over every layer contribution | 🟡 small-scale form |
| Stable LatentMoE | 896 routed, top-16, 2 shared, latent width 3,584 | 8 routed, top-2, 1 shared, latent width 64 | 🟡 scaled down |
| First dense layer | `first_k_dense_replace=1`, then 92 MoE layers | `dense_layers=1`, then MoE blocks | 🟢 pattern |
| Quantile Balancing | Global histogram estimate, applied next batch | Exact local-batch quantile, applied next forward | 🟡 scaled down |
| SiTU-GLU | β₁=4, β₂=25 | β₁=4, β₂=25 | 🟢 equation |
| Optimizer | Per-Head Muon with QK-clip; 1% warmup, cosine decay, weight decay 0.1 | Per-head Muon on matrices, AdamW tail, same schedule shape; no QK-clip | 🟡 scaled down |
| Decoding | Fused kernels; constant KDA state, compressed MLA cache | Same state-versus-cache split in a plain prefill + step loop | 🟢 pattern |
| Context and tokens | 1,048,576 learned-token context | 128 raw bytes by default | 🟡 teaching scale |
| Embeddings | Untied input and output embeddings | One tied byte embedding and head | 🟡 scaled down |
| Vision | 27-layer, 401M MoonViT-V2: RMSNorm, no bias terms, trained from scratch, 2×2 pixel shuffle into an MLP projector | 2-layer tower with the same shape rules and pixel shuffle, native-resolution packing, 2D RoPE, one projection matrix; `--vision` | 🟡 small-scale form |
| Native quantization | MXFP4 routed-expert weights, MXFP8 input activations, QAT across post-training | Same formats, block scales and placement, straight-through QAT from step one, simulated in float32; `--quantize` | 🟡 same formats, simulated |

The KDA path deliberately omits ShortConv, Swish projections, low-rank decay projection, and the
chunkwise fused algorithm. The global layer is not MLA. Quantile Balancing is exact only over the
local teaching batch, not a distributed global histogram. The vision tower reads small procedural
pictures rather than photographs and has no temporal path, so video is still absent. Quantization
is simulated in float32 instead of executed by a low-precision kernel: the arithmetic is the
format's, the speed is not. These boundaries are explicit; tests cover the recurrence’s numerical
behavior, causality, cached-decoding equivalence, routing counts, next-step bias update, optimizer
parameter grouping, initialization scale, valid configuration, generation guardrails,
corpus-window boundaries, packed multi-resolution encoding, image-prefix alignment, and the
microscaling grid down to its tie-breaking rule and packed bit layout.

One subtle experiment: with `top_k=1`, the normalized selected router weight is exactly one, so the
router receives essentially no gradient through mixture weights. K3 uses top-16. Treat top-1 here
as a demonstration of that failure mode, not as a recommended setting.

## Pictures and four-bit experts

The last two rows of that table used to read “out of scope.” Both now have a small-scale version,
and both are off by default:

```bash
microk3 --vision --steps 500      # train the patch tower to name shapes
microk3 --quantize --steps 200    # MXFP4 routed experts, MXFP8 input activations
```

`--vision` builds `MicroMoonViT`: patches at each image's own resolution, 2D RoPE over patch
coordinates, and a block-diagonal mask so several differently sized pictures share one packed
attention pass—the report's intra-frame spatial pass, with one frame per sample. A 2×2 pixel
shuffle then folds four patches into one token before a single projection into the byte stream,
which is where the token count actually gets paid for: a 40-pixel picture costs 100 patches but
only 25 tokens. Those tokens are a prefix of the byte sequence and ride the same depth bus, KDA
states, and global caches as text, so there is one backbone and no alignment stage. Training draws
squares, circles, triangles, and crosses at 24, 32, or 40 pixels and asks for the name; the last
image token predicts the caption's first byte, so nothing but the picture chooses the word. At the
default seed, 500 steps reads 7 of 8 freshly drawn pictures correctly. That is the whole claim—the
mechanism works at a scale where you can watch it, not that this tower sees anything.

`--quantize` puts the deployment precision in the training loop. MXFP4 keeps each weight as an
E2M1 element with one E8M0 power-of-two scale shared by a block of 32; MXFP8 does the same with
E4M3 elements for the input activations. Rounding is the format's round-half-to-even, so exact
midpoints land on even codes; `pack_mxfp4` writes the real bit layout, two 4-bit codes per byte
plus one exponent byte per block, and `unpack_mxfp4` returns exactly what training saw. A
straight-through estimator keeps the gradient path intact, so the weights learn where the grid is
rather than being rounded onto it afterwards. Only the routed experts quantize—the router, the
latent projections, the shared expert, and every attention matrix stay in float, as §4.1.4
describes. At the default shape that covers 884,736 of 1,645,496 weights, which pack into 470,016
bytes against 3,538,944 in float32: 4.25 bits per weight, and a reminder that experts are where
the memory lives.

Four-bit experts cost little at this size—little enough that the seed matters more than the format
does. Two hundred steps on the bundled corpus end between 0.041 and 0.053 with `--quantize` and
between 0.045 and 0.057 without, across seeds 0–3 and 42 on CPU. The spread within either setting
is roughly six times the gap between their means, and `--quantize` finishes lower at three of those
five seeds. Read that as a smoke test rather than a scaling result: the bundled corpus is short
enough to memorize, and a model this small has few weights whose precision is doing real work. If
you want a number that means something here, sweep `--seed` and compare distributions, not runs.

## Report-to-code map

| Report section | Read this code | Preserved | Deliberately omitted or reduced |
|---|---|---|---|
| §2.1.1, Eqs. 1 & 5 | `KimiDeltaAttention` | Decay-before-correction recurrence, channel-wise retention, learned per-head scale | ShortConv, Swish, low-rank decay projection, chunkwise kernel |
| §2.1.1, Eq. 6 | `KimiDeltaAttention.forward` | Head RMSNorm and full-rank sigmoid output gate | Fused training kernel |
| §2.2, Eqs. 8–10 | `MicroK3._mix_depth` and `Block.forward` | Learned pseudo-query, normalized keys, softmax over prior contributions | Block grouping and intra-block partial sums |
| §2.3, Eqs. 11–12 | `StableLatentMoE` and `SiTUGLU` | Latent routed path, pre-up RMSNorm, bounded GLU, shared path | Report-scale widths and second shared expert |
| §2.3.3, Eqs. 13–14 | `StableLatentMoE._update_router_bias` | Bias only affects dispatch; quantile bias is used on the next pass | Distributed histogram approximation |
| §2.4 | `MicroMoonViT`, `prefix_targets`, and `forward`'s `prefix` | Native-resolution patches, RMSNorm, bias-free projections, 2×2 pixel shuffle, one shared backbone trained by next-token prediction | Video, temporal attention and pooling, the MLP projector, 27-layer scale |
| §2.5 and §3.3 | `orthogonalize`, `Muon`, `build_optimizers` | Per-head Newton–Schulz step, matrix/tail split, 1% warmup + cosine shape | QK-clip, distributed sharding, report-scale tuning |
| §4.1.4 | `quantize_mx`, `mx_linear`, `pack_mxfp4` | E2M1 and E4M3 element grids, E8M0 block scales, routed-experts-only placement, QAT with a straight-through estimator | Low-precision kernels, the post-training-only schedule, RL rollout sharing |

Default tensor shapes make the scale reduction concrete:

```text
tokens                         [batch, time]
hidden                         [batch, time, 128]
one KDA recurrent state        [batch, 4 heads, 32 key channels, 32 value channels]
MoE router scores              [batch, time, 8 experts]
selected experts               [batch, time, 2]
depth sources                  embedding + one contribution per completed layer
```

## Model archaeology without a 1.5 TB accident

Moonshot AI announced K3 on July 16, 2026; the public repository and technical report followed on
July 27. The metadata helper lists remote filenames and performs unauthenticated HEAD requests for
weight sizes; it has no file-download call and writes nothing:

```bash
pip install -e '.[inspect]'
python scripts/inspect_k3_metadata.py
```

On the 2026-07-28 UTC snapshot, it observed 118 files, including 96 weight shards totaling 1.420
TiB. Remote repositories can change, so the script prints the current values rather than treating
that snapshot as permanent. See [`docs/report-notes.md`](docs/report-notes.md) for primary links,
equation-level notes, and the exact simplifications.

## Modal

Use credits deliberately. The cloud entrypoint has a 30-minute hard timeout, caps the step count,
uses one GPU, and creates no persistent model-weight volume.

```bash
pip install -e '.[modal]'
modal setup
modal run modal_train.py --steps 500
```

It defaults to an L40S and the bundled corpus. `--batch-size`, `--block-size`, `--optimizer`,
`--vision`, and `--quantize` pass through, and `--data path/to/tiny.txt` ships a corpus of up to
8 MiB with the run; anything larger belongs in a Modal Volume. Cloud runs default to
`--generate 0`; opt in with `--generate 120` only when the corpus is non-sensitive and you want
sampled text in the run logs. `--data` cannot be combined with `--vision`, which draws its own
procedural pictures. Do not upload private or confidential corpora. Start at 100–500 steps,
inspect Modal's live cost dashboard, then scale consciously. **A free-credit balance is not a
spending guarantee**; pricing and availability change, so check Modal before launching. The K3
weights are never fetched.

## Experiments worth trying

- Plot `block.ffn.last_load` and `block.ffn.router_bias` for an MoE block; which experts
  specialize, and how quickly does the local Quantile Balancing update respond?
- Disable `_update_router_bias` for one run and compare expert loads.
- Change `top_k` from 2 → 1 and confirm why the normalized router-weight gradient disappears.
- Replace the bounded decay with an unbounded softplus decay and inspect long-prefix retention.
- Add the omitted ShortConv + Swish projections, then compare the tensor trace.
- Train the same seed with `--optimizer adamw` and compare loss curves. At a matched learning rate
  the two finish close together on this corpus; what Muon actually buys here is insensitivity to
  that learning rate, so sweep `--muon-lr` from 0.005 to 0.05 and watch how little moves.
- Raise and lower `--learning-rate` in `--optimizer muon` runs. It touches only the embedding and
  gain tail, yet it moves the loss more than `--muon-lr` does: the tied byte head is the bottleneck.
- Generate far past `--block-size` and watch the global caches grow while every KDA state stays
  the same size.
- Compress the global layers' plain KV cache into a small latent, as MLA does, and measure memory.
- K3 turns quantization on for post-training only. Flip `cfg.mx_qat` partway through a run instead
  of at step zero, and watch how much loss the switch costs when the weights were not expecting it.
- Point `mx_linear` at the shared expert or the attention projections too. The report leaves them
  in higher precision; the loss curve shows why that is not just caution.
- Delete the pixel shuffle and project each patch on its own. The captions barely change, and the
  prefix gets four times longer—which at 1M tokens is the entire argument for it.
- Feed the tower two resolutions in one call and compare against encoding each alone. They match,
  because the packed mask is the only thing keeping images apart.

## Development

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

CI runs those checks on Python 3.10 and 3.13, builds the wheel and source distribution, installs
the wheel outside the checkout, imports `microk3`, and exercises the installed CLI.

## Scope, sources, and license

Architecture facts come from Moonshot AI's [official repository](https://github.com/MoonshotAI/Kimi-K3)
and [technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf).
This repository contains original educational code under MIT. Kimi K3 weights have their own
[model license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE); this project neither
redistributes nor relicenses them. Contributions that make an idea clearer are especially welcome.
