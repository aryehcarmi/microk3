# microK3

[![CI](https://github.com/aryehcarmi/microk3/actions/workflows/ci.yml/badge.svg)](https://github.com/aryehcarmi/microk3/actions/workflows/ci.yml)

A small, readable model for learning the ideas behind Kimi K3: recurrent Kimi Delta
Attention, periodic global attention, Attention Residuals across depth, and a Stable
LatentMoE across width. Two further pieces of the release, the MoonViT-V2 vision tower
and MXFP4 deployment precision, are included at the same scale behind their own flags.
The whole model is one file, [`microk3.py`](microk3.py), with a diagram and a
regression suite.

microK3 is K3-inspired, not Kimi K3. It is a ~1.65M-parameter teaching model, not a
reproduction or a distillation, and it never downloads the 2.8T-parameter weights. The
official implementation needs industrial infrastructure; this repository keeps only the
conceptual spine, at a scale where you can read and run all of it.

![microK3 on one page: the forward pass over a depth bus, the KDA recurrence, the
depth-attention pattern, Stable LatentMoE routing, decoding memory, the optimizer split,
and the scale gap to Kimi K3](assets/architecture.svg)

## Quickstart

```bash
git clone https://github.com/aryehcarmi/microk3
cd microk3
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
microk3 --steps 20
```

On CPU-only Linux, install torch from the CPU index first so pip does not pull the much
larger CUDA build:

```bash
pip install 'torch>=2.4.1,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
```

The tests should report `65 passed`. The first training loss should be near the uniform
byte-token baseline `ln(256) ≈ 5.55`; an initial loss in the tens or hundreds is a bug.
Twenty steps is enough to watch the loss fall and nothing more: the sample printed at
the end is byte noise, and stays noise until a few hundred steps on a real corpus. CPU
is fine at this scale. `--device mps` trains slightly faster and samples slower, because
the KDA token loop launches many small kernels.

To train on your own text, code, or arbitrary bytes:

```bash
microk3 --data path/to/tiny.txt --steps 1000
```

The file must contain at least `block_size + 1` bytes (129 at the default setting); a
larger corpus is much more useful. `--block-size 64` speeds up CPU experiments. Training
prints a sample at the end and writes no checkpoint; use `--generate 0` for a train-only
run. The default optimizer is per-head Muon with a warmup-then-cosine schedule, and
`--optimizer adamw` switches back. `--vision` and `--quantize` enable the two optional
parts described below. `microk3 --help` lists the batch size, learning rate, prompt,
temperature, and device controls.

Model shape (`layers`, `dim`, `experts`, `top_k`, `dense_layers`, the `vision_*` widths)
is deliberately not exposed as CLI flags: edit the `Config` defaults at the top of
`microk3.py` so every change stays visible in the file you are reading. `--vision` and
`--quantize` only decide whether a part is built at all.

## Reading order

Read [`microk3.py`](microk3.py) top to bottom:

1. `KimiDeltaAttention`: Eq. 1 as a recurrent token loop, with channel-wise decay before
   the delta correction; Eq. 5 supplies the bounded log-decay.
2. `GatedAttention`: periodic causal global attention. K3 uses compressed MLA; microK3
   uses ordinary attention so the lesson fits on one screen.
3. `SiTUGLU` and `StableLatentMoE`: route through a small latent, normalize the routed
   aggregate, and update dispatch biases with an exact batch-local version of Quantile
   Balancing.
4. `MicroK3._mix_depth`: AttnRes-style retrieval over the embedding and individual layer
   contributions, with a separate final-output query.
5. `LayerCache` and `MicroK3.generate`: prefill once, then decode step by step. KDA
   states stay the same size forever; only the global layers' caches grow with the
   sequence.
6. `orthogonalize`, `Muon`, and `build_optimizers`: K3's per-head orthogonalized step
   for matrix weights, with an undecayed AdamW tail for embeddings, gains, and biases.
7. `MXFormat`, `quantize_mx`, `mx_linear`, and `pack_mxfp4`: the deployment precision,
   from the element grid and its shared block scale to the packed bit layout.
8. `rope_2d`, `MicroMoonViT`, and `prefix_targets`: patches at the image's own
   resolution, several pictures packed into one attention pass, then projected into the
   byte stream as a prefix.

## What matches K3, and what doesn't

| Idea | Kimi K3 | microK3 | Status |
|---|---|---|---|
| Hybrid schedule | 69 KDA + 24 Gated MLA; 3:1 blocks plus final global layer | 3 KDA + 1 global by default; arbitrary depths still end globally | same pattern |
| Delta recurrence | Channel-wise decay, then delta correction | Same recurrence in a readable token loop | same equation |
| KDA projections | ShortConv + Swish Q/K/V; low-rank decay logits | Plain linear Q/K/V; full-rank decay logits | simplified |
| Lower-bounded decay | `g = -5 sigmoid(exp(A) z)` per key channel | Same mapping, per key channel | same equation |
| KDA execution | Fused chunkwise kernel | Sequential Python loop | same recurrence, slow loop |
| Global attention | Gated MLA, NoPE | Gated MHA, no positional embeddings | substituted |
| Attention Residuals | Eight block-level groups with partial sums | Full attention over every layer contribution | scaled down |
| Stable LatentMoE | 896 routed, top-16, 2 shared, latent width 3,584 | 8 routed, top-2, 1 shared, latent width 64 | scaled down |
| First dense layer | `first_k_dense_replace=1`, then 92 MoE layers | `dense_layers=1`, then MoE blocks | same pattern |
| Quantile Balancing | Global histogram estimate, applied next batch | Exact local-batch quantile, applied next forward | scaled down |
| SiTU-GLU | β₁=4, β₂=25 | β₁=4, β₂=25 | same equation |
| Optimizer | Per-Head Muon with QK-clip; 1% warmup, cosine decay, weight decay 0.1 | Per-head Muon on matrices, AdamW tail, same schedule shape; no QK-clip | scaled down |
| Decoding | Fused kernels; constant KDA state, compressed MLA cache | Same state-versus-cache split in a plain prefill + step loop | same pattern |
| Context and tokens | 1,048,576 learned-token context | 128 raw bytes by default | scaled down |
| Embeddings | Untied input and output embeddings | One tied byte embedding and head | scaled down |
| Vision | 27-layer, 401M MoonViT-V2: RMSNorm, no bias terms, trained from scratch, 2×2 pixel shuffle into an MLP projector | 2-layer tower with the same shape rules and pixel shuffle, native-resolution packing, 2D RoPE, one projection matrix; `--vision` | scaled down |
| Native quantization | MXFP4 routed-expert weights, MXFP8 input activations, QAT across post-training | Same formats, block scales and placement, straight-through QAT from step one, simulated in float32; `--quantize` | simulated |

The KDA path omits ShortConv, Swish projections, the low-rank decay projection, and the
chunkwise fused algorithm. The global layer is not MLA. Quantile Balancing is exact only
over the local batch, not a distributed global histogram. The vision tower reads small
procedural pictures rather than photographs and has no temporal path, so video is
absent. Quantization is simulated in float32 rather than executed by a low-precision
kernel: the arithmetic is the format's, the speed is not. See
[`docs/report-notes.md`](docs/report-notes.md) for the full list with sources.

One subtle experiment: with `top_k=1`, the normalized selected router weight is exactly
one, so the router receives essentially no gradient through mixture weights. K3 uses
top-16. Treat top-1 here as a demonstration of that failure mode, not as a setting.

## Vision and quantization

Both parts are off by default:

```bash
microk3 --vision --steps 500      # train the patch tower to name shapes
microk3 --quantize --steps 200    # MXFP4 routed experts, MXFP8 input activations
```

`--vision` builds `MicroMoonViT`: patches at each image's own resolution, 2D RoPE over
patch coordinates, and a block-diagonal mask so several differently sized pictures share
one packed attention pass. A 2×2 pixel shuffle then folds four patches into one token
before a single projection into the byte stream, which is where the token count gets
paid for: a 40-pixel picture costs 100 patches but only 25 tokens. Those tokens are a
prefix of the byte sequence and ride the same depth bus, KDA states, and global caches
as text, so there is one backbone and no alignment stage. Training draws squares,
circles, triangles, and crosses at 24, 32, or 40 pixels and asks for the name; the last
image token predicts the caption's first byte, so only the picture chooses the word. At
the default seed, 500 steps reads 7 of 8 freshly drawn pictures correctly.

`--quantize` puts the deployment precision in the training loop. MXFP4 stores each
weight as an E2M1 element with one E8M0 power-of-two scale shared by a block of 32;
MXFP8 does the same with E4M3 elements for the input activations. Rounding is the
format's round-half-to-even, `pack_mxfp4` writes the real bit layout (two 4-bit codes
per byte plus one exponent byte per block), and `unpack_mxfp4` returns exactly what
training saw. A straight-through estimator keeps the gradient path intact, so the
weights learn where the grid is rather than being rounded onto it afterwards. Only the
routed experts quantize; the router, the latent projections, the shared expert, and
every attention matrix stay in float, as §4.1.4 describes. At the default shape that
covers 884,736 of 1,645,496 weights, which pack into 470,016 bytes against 3,538,944 in
float32, or 4.25 bits per weight.

Quantization costs little at this size. Two hundred steps on the bundled corpus end
between 0.041 and 0.053 with `--quantize` and between 0.045 and 0.057 without, across
seeds 0–3 and 42 on CPU. The spread within either setting is roughly six times the gap
between their means, so single runs are smoke tests; to compare the settings, sweep
`--seed` and compare distributions.

## Report-to-code map

| Report section | Read this code | Preserved | Omitted or reduced |
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

## Inspecting the real K3

Moonshot AI announced K3 on July 16, 2026; the public repository and technical report
followed on July 27. A helper script lists the remote weight repository's filenames and
performs unauthenticated HEAD requests for sizes. It has no file-download call and
writes nothing:

```bash
pip install -e '.[inspect]'
python scripts/inspect_k3_metadata.py
```

On the 2026-07-28 UTC snapshot it observed 118 files, including 96 weight shards
totaling 1.420 TiB. Remote repositories can change, so the script prints current values
rather than treating that snapshot as permanent. See
[`docs/report-notes.md`](docs/report-notes.md) for primary links, equation-level notes,
and the exact simplifications.

## Modal

The cloud entrypoint has a 30-minute hard timeout, caps the step count, uses one GPU,
and creates no persistent model-weight volume:

```bash
pip install -e '.[modal]'
modal setup
modal run modal_train.py --steps 500
```

It defaults to an L40S and the bundled corpus. `--batch-size`, `--block-size`,
`--optimizer`, `--vision`, and `--quantize` pass through, and `--data path/to/tiny.txt`
ships a corpus of up to 8 MiB with the run; anything larger belongs in a Modal Volume.
Cloud runs default to `--generate 0`; opt in with `--generate 120` only when you want
sampled text in the run logs, and do not upload private corpora. `--data` cannot be
combined with `--vision`, which draws its own pictures. Start at 100–500 steps and check
Modal's cost dashboard and current pricing before scaling up. The K3 weights are never
fetched.

## Experiments worth trying

- Plot `block.ffn.last_load` and `block.ffn.router_bias` for an MoE block; which experts
  specialize, and how quickly does the Quantile Balancing update respond?
- Disable `_update_router_bias` for one run and compare expert loads.
- Change `top_k` from 2 to 1 and confirm why the router-weight gradient disappears.
- Replace the bounded decay with an unbounded softplus decay and inspect long-prefix
  retention.
- Add the omitted ShortConv + Swish projections, then compare the tensor trace.
- Train the same seed with `--optimizer adamw` and compare loss curves. At a matched
  learning rate the two finish close together on this corpus; what Muon buys here is
  insensitivity to that rate, so sweep `--muon-lr` from 0.005 to 0.05 and watch how
  little moves.
- Raise and lower `--learning-rate` in `--optimizer muon` runs. It touches only the
  embedding and gain tail, yet it moves the loss more than `--muon-lr` does, because the
  tied byte head is the bottleneck.
- Generate far past `--block-size` and watch the global caches grow while every KDA
  state stays the same size.
- Compress the global layers' KV cache into a small latent, as MLA does, and measure
  memory.
- K3 turns quantization on for post-training only. Flip `cfg.mx_qat` partway through a
  run instead of at step zero, and measure what the switch costs when the weights were
  not expecting it.
- Point `mx_linear` at the shared expert or the attention projections too, and see on
  the loss curve why the report leaves them in higher precision.
- Delete the pixel shuffle and project each patch on its own. The captions barely
  change, but the prefix gets four times longer, which at a 1M-token context is the
  argument for it.
- Feed the tower two resolutions in one call and compare against encoding each alone.
  They match, because the packed mask is the only thing keeping images apart.

## Development

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

CI runs those checks on Python 3.10 and 3.13, builds the wheel and source distribution,
installs the wheel outside the checkout, imports `microk3`, and exercises the installed
CLI.

## Sources and license

Architecture facts come from Moonshot AI's
[official repository](https://github.com/MoonshotAI/Kimi-K3) and
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf).
This repository contains original educational code under MIT. Kimi K3 weights have
their own [model license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE);
this project neither redistributes nor relicenses them.
