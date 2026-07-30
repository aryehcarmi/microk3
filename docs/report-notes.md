# Reading Kimi K3 without hand-waving

These notes separate **reported fact**, **microK3's implementation choice**, and **unknown**.
Primary sources are Moonshot AI's [Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3),
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf), released
weights [configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json), and
[modeling code](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py).
They were checked on 2026-07-28 UTC.

## Reported architecture

- 2.8T total parameters and 104B activated; 93 layers (one dense), width 7,168, and 96 heads.
- 69 KDA and 24 Gated MLA layers. A 3 KDA + 1 MLA pattern repeats, followed by an additional final
  Gated MLA layer; vocabulary is 160K and context is 1,048,576 tokens. MLA layers use no positional
  encoding.
- Stable LatentMoE projects routed work to width 3,584, has 896 routed experts with 16 selected,
  expert hidden width 3,072, and two full-width shared experts.
- KDA has a recurrent state for each head. Retention is channel-wise across the key dimension;
  Q/K are L2-normalized; and the output has a full-rank sigmoid gate.
- KDA applies ShortConv + Swish to Q/K/V projections. Its readable recurrence first decays the old
  state, then predicts from that decayed state, then applies the delta correction.
- Eq. 5 maps each decay logit with `g = -5 sigmoid(exp(A) z)` and `alpha = exp(g)`, where `A` is a
  learned per-head log-scale and each `z` is per key channel.
- Block AttnRes attends over RMS-normalized earlier block representations with learned depth
  pseudo-queries. K3 has eight layer blocks: seven full 12-layer blocks and one partial 9-layer
  block. Counting the embedding gives nine block-level sources.
- SiTU-GLU soft-caps gate and up branches with `beta * tanh(x / beta)`; beta values are 4 and 25.
- Input and output embeddings are untied (`tie_word_embeddings: false`).
- Quantile Balancing adds per-expert biases only to Top-k dispatch. Mixture weights use unbiased
  sigmoid scores. The next-step bias is the negative `(1 - k/n)` quantile of each expert's margin
  over the current Top-(k+1) cutoff, centered to zero mean.
- Matrices train with Per-Head Muon: Newton–Schulz orthogonalization applied per attention head's
  block, keeping K2's weight-clipping mechanism, with cosine decay after a 1% linear warmup and
  weight decay 0.1. Peak learning rates, batch sizes, and total token counts are not published.
- §2.4: the 401M-parameter, 27-layer MoonViT-V2 is trained from scratch with next-token
  prediction, a departure from the SigLIP-initialized encoders of Kimi K2.5 that the report
  attributes to training stability. It uses RMSNorm and removes every bias term from its linear
  and attention projections. Images and videos share all parameters; attention is factorized into
  intra-frame spatial and inter-frame temporal passes, with temporal pooling along time. A 2×2
  pixel shuffle cuts the visual token count fourfold before a lightweight MLP projector maps the
  encoder output into the LLM, keeping inputs up to 3,584×3,584 pixels affordable. Text, images,
  and video share one backbone and one context, with no post-hoc modality-alignment stage.
- §4.1.4: MoE expert weights are quantized to MXFP4 with activations computed in MXFP8, while all
  non-expert components—attention projections, latent MoE projections, shared experts, and MoE
  routers—stay in higher precision. Quantization-aware training runs throughout post-training,
  covering both SFT and RL, and rollout and training share the scheme during RL. This is not
  post-training quantization of the whole model.
- The element and scale formats themselves are not defined in the K3 report; it cites
  *Microscaling Data Formats for Deep Learning* (arXiv:2310.10537), where MXFP4 is E2M1, MXFP8 is
  E4M3 or E5M2, and each block of 32 values along the reduction axis shares one E8M0 scale.

## What the tiny code preserves

`microk3.py` preserves:

- the Eq. 1 order of operations: channel-wise decay before the delta prediction and correction;
- the Eq. 5 bounded log-decay, including learned per-head `exp(A)`;
- head RMSNorm and the full-rank sigmoid output gate;
- a 3:1 hybrid schedule with a guaranteed global final layer;
- AttnRes-style retrieval over the embedding and separate layer contributions;
- a latent routed path, pre-up RMSNorm, a full-width shared path, and the SiTU constants;
- bias-separated routing weights and an exact, batch-local next-step Quantile Balancing update;
- a dense first layer before the routed stack;
- per-head Muon on matrix weights, an undecayed AdamW tail, and the warmup-then-cosine schedule
  shape;
- decoding that carries a fixed-size KDA state while only global layers append keys and values;
- a vision tower with RMSNorm, no bias terms, per-image resolution, a 2×2 pixel shuffle before
  projection, and training from scratch by next-token prediction inside one shared backbone;
- the MX element grids down to their round-half-to-even tie rule, one E8M0 scale per block of 32,
  MXFP4 for routed-expert weights with MXFP8 for their input activations, and a packed layout of
  two 4-bit codes per byte plus one exponent byte per block.

It deliberately changes:

- ShortConv + Swish Q/K/V projections to plain linear projections;
- the low-rank decay-logit projection to one plain full-rank projection;
- the fused chunkwise KDA algorithm to a sequential token loop with the same recurrence;
- Gated MLA to ordinary gated causal multi-head attention;
- eight block-level AttnRes groups to full attention over four individual layer contributions;
- distributed histogram Quantile Balancing to `torch.quantile` on the current local batch;
- 896/top-16/two-shared routing to 8/top-2/one-shared routing;
- QK-clip and the distributed Muon implementation to a bare batched Newton–Schulz step;
- unpublished report-scale learning rates and batch sizes to small fixed defaults;
- untied input and output embeddings to one tied byte embedding and head;
- learned subword tokens to raw bytes;
- a 27-layer, 401M-parameter tower on real images to a 2-layer tower on procedural shapes at 24 to
  40 pixels, with the lightweight MLP projector reduced to one bias-free matrix;
- MXFP8's element type, which the report does not name, to E4M3; the specification also allows
  E5M2, and which one K3 computes with is unknown from these sources;
- quantization-aware training across SFT and RL to quantization from the first pre-training step,
  since there is no post-training stage here to switch it on for;
- executed low-precision arithmetic to float32 simulation with a straight-through estimator: the
  rounding is the format's, the speed is not.

It omits video and the temporal half of the vision pathway, million-token execution, MTP,
speculative decoding, distributed training, and the report's infrastructure work.

Two vision details are microK3's own, not the K3 report's. §2.4 does not say how MoonViT-V2 handles
varying resolutions, so packing images into one sequence behind a block-diagonal mask, and 2D RoPE
over patch coordinates, are implementation choices here. Both come from the earlier MoonViT as
described in the [Kimi-VL report](https://arxiv.org/abs/2504.07491), which flattens patches into 1D
sequences in NaViT's style and applies 2D RoPE across height and width. Whether V2 keeps either
mechanism is unknown from these sources. What the K3 report does state—one shared backbone,
per-image resolution up to 3,584×3,584, and an intra-frame spatial attention pass—is what the
block-diagonal mask is built to satisfy.

## Weight inspection status

The official weight repository is `moonshotai/Kimi-K3`. On 2026-07-28 UTC,
`scripts/inspect_k3_metadata.py` observed 118 remote files, including 96 recognized weight shards
whose HEAD metadata summed to 1.420 TiB. It downloaded no file bodies.

Those counts are a dated observation, not an architectural constant. The helper lists the current
repository, performs an unauthenticated HEAD metadata request only for recognized weight suffixes,
and has no download call. Readers can inspect its short source before running it:

```bash
pip install -e '.[inspect]'
python scripts/inspect_k3_metadata.py
```

The script does not inspect tensors, verify tensor names, or execute remote modeling code. Claims
about tensor-level structure should therefore be checked against the released config and modeling
source rather than inferred from filenames.
