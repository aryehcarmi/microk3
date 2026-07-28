# Reading Kimi K3 without hand-waving

These notes distinguish **reported fact**, **our implementation choice**, and **unknown**. Primary
source: Moonshot AI's [Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3) and its
[technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf), accessed
2026-07-28.

## Reported architecture

- 2.8T total parameters and 104B activated; 93 layers (one dense), width 7,168, 96 heads.
- 69 KDA and 24 Gated MLA layers, generally repeating three KDA then one MLA; vocabulary 160K and
  context 1,048,576. MLA layers use no positional encoding.
- Stable LatentMoE projects routed work to width 3,584, has 896 routed experts with 16 selected,
  expert hidden width 3,072, and two full-width shared experts.
- KDA uses channel-wise recurrent state. Its log decay is `-5 sigmoid(.)`, hence each retention
  factor is in `(e^-5, 1)`. Q/K are L2-normalized and the output has a full-rank sigmoid gate.
- Block AttnRes attends over RMS-normalized earlier block representations with learned depth
  pseudo-queries. K3 uses eight 12-layer blocks plus a partial block.
- SiTU-GLU soft-caps gate and up branches with `β tanh(x/β)`; β values are 4 and 25.
- Quantile Balancing uses per-expert biases only for top-k dispatch. Normalized mixture weights use
  unbiased sigmoid router scores; a global quantile estimates a next-step bias targeting equal load.
- The 401M-parameter, 27-layer MoonViT-V2 is trained from scratch with next-token prediction.
- The release reports MXFP4 routed-expert weights and MXFP8 input activations from quantization-aware
  training. This is not post-training quantization of the whole model.

## What the tiny code changes

`microk3.py` preserves bounded decay, output gating, SiTU constants, latent routing, bias-separated
dispatch, and depth mixing. It substitutes standard causal MHA for MLA, uses a direct recurrent loop
instead of the report's chunkwise fused KDA formulation, attends over every prior layer rather than
blocks, uses one shared expert, and omits vision, quantization, short convolution, MTP, distributed
training, and the QB bias update.

## Weight inspection status

The official weight repository is `moonshotai/Kimi-K3`. We intentionally did not download shards.
Hugging Face returned HTTP 403 in the development environment, so exact config keys, tensor names,
shard byte totals, and modeling-code details were **not verified from weights**. No such details are
claimed here. The metadata-only helper is included so readers can safely repeat that inspection in
an environment with access.
