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
- Quantile Balancing adds per-expert biases only to Top-k dispatch. Mixture weights use unbiased
  sigmoid scores. The next-step bias is the negative `(1 - k/n)` quantile of each expert's margin
  over the current Top-(k+1) cutoff, centered to zero mean.
- Matrices train with Per-Head Muon: Newton–Schulz orthogonalization applied per attention head's
  block, keeping K2's weight-clipping mechanism, with cosine decay after a 1% linear warmup and
  weight decay 0.1. Peak learning rates, batch sizes, and total token counts are not published.
- The 401M-parameter, 27-layer MoonViT-V2 is trained from scratch with next-token prediction.
- The release reports MXFP4 routed-expert weights and MXFP8 input activations from
  quantization-aware training. This is not post-training quantization of the whole model.

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
- decoding that carries a fixed-size KDA state while only global layers append keys and values.

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
- learned subword tokens to raw bytes.

It omits vision, native quantization, million-token execution, multimodal projection, MTP,
distributed training, and the report's infrastructure work.

## Weight inspection status

The official weight repository is `moonshotai/Kimi-K3`. On 2026-07-28 UTC,
`scripts/inspect_k3_metadata.py` observed 118 remote files, including 96 recognized weight shards
whose HEAD metadata summed to 1.420 TiB. It downloaded no file bodies.

Those counts are a dated observation, not an architectural constant. The helper lists the current
repository, performs a HEAD metadata request only for recognized weight suffixes, and has no
download call. Readers can inspect its short source before running it:

```bash
pip install -e '.[inspect]'
python scripts/inspect_k3_metadata.py
```

The script does not inspect tensors, verify tensor names, or execute remote modeling code. Claims
about tensor-level structure should therefore be checked against the released config and modeling
source rather than inferred from filenames.
