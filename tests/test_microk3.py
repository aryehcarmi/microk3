import argparse
import math
import sys

import pytest
import torch

from microk3 import (
    MXFP4,
    MXFP8,
    SHAPES,
    Config,
    GatedAttention,
    KimiDeltaAttention,
    LayerCache,
    MicroK3,
    SiTUGLU,
    StableLatentMoE,
    apply_rope,
    available_device,
    build_optimizers,
    caption,
    delta_rule_step,
    fake_quantize_mx,
    learning_rate_scale,
    main,
    mx_shared_scale,
    mxfp4_footprint,
    orthogonalize,
    pack_mxfp4,
    parameter_groups,
    prefix_targets,
    quantize_mx,
    rope_2d,
    sample_batch,
    shapes_batch,
    terminal_text,
    unpack_mxfp4,
)


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "dim": 16,
        "latent_dim": 8,
        "expert_hidden": 12,
        "layers": 4,
        "heads": 2,
        "experts": 4,
        "top_k": 2,
        "block_size": 8,
    }
    return Config(**(values | overrides))


def tiny_vision_config(**overrides):
    """A tower small enough for tests: 16 patches per 16x16 image, 4 tokens after pixel shuffle.

    Captions are ordinary ASCII bytes, so these configurations keep the full byte vocabulary.
    """
    values = {
        "vocab_size": 256,
        "vision_layers": 2,
        "vision_dim": 16,
        "vision_heads": 2,
        "vision_hidden": 12,
        "patch_size": 4,
    }
    return tiny_config(**(values | overrides))


def test_forward_backward_and_routing():
    torch.manual_seed(0)
    model = MicroK3(tiny_config())
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None
    )
    assert isinstance(model.blocks[0].ffn, SiTUGLU)
    for block in model.blocks[1:]:
        assert block.ffn.last_load.sum() == x.numel() * 2


def test_initial_loss_and_logits_are_well_scaled():
    torch.manual_seed(0)
    model = MicroK3(tiny_config()).eval()
    x = torch.randint(0, 32, (4, 8))
    y = torch.randint(0, 32, (4, 8))
    with torch.no_grad():
        logits, loss = model(x, y)
    assert logits.std() < 1.0
    assert abs(loss.item() - math.log(32)) < 1.0


def test_tiny_model_can_overfit_one_batch():
    torch.manual_seed(0)
    model = MicroK3(tiny_config())
    x = torch.randint(0, 32, (2, 8))
    y = torch.roll(x, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def test_muon_overfits_one_batch():
    torch.manual_seed(0)
    model = MicroK3(tiny_config())
    x = torch.randint(0, 32, (2, 8))
    y = torch.roll(x, shifts=-1, dims=1)
    optimizers = build_optimizers(model, "muon", learning_rate=1e-2)
    losses = []
    for _ in range(10):
        _, loss = model(x, y)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def test_orthogonalize_pushes_singular_values_toward_one():
    torch.manual_seed(0)
    for shape in ((2, 8, 4), (2, 4, 8)):
        values = torch.linalg.svdvals(orthogonalize(torch.randn(shape)))
        assert ((values > 0.5) & (values < 1.5)).all()


@pytest.mark.parametrize("config", [tiny_config(), tiny_vision_config(mx_qat=True)])
def test_parameter_groups_cover_every_parameter_exactly_once(config):
    model = MicroK3(config)
    matrices, others = parameter_groups(model)
    grouped = [parameter for parameters in matrices.values() for parameter in parameters] + others
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in model.parameters()}


def test_learning_rate_scale_warms_up_then_decays_to_zero():
    scales = [learning_rate_scale(step, 200) for step in range(200)]
    assert scales[0] < 1.0
    assert max(scales) == 1.0
    assert scales[-1] < 1e-3


def test_situ_glu_is_finite_for_extreme_inputs():
    layer = SiTUGLU(4, 8)
    assert torch.isfinite(layer(torch.full((2, 4), 1e6))).all()


def test_half_precision_zero_qk_stays_finite():
    layer = KimiDeltaAttention(tiny_config()).half()
    with torch.no_grad():
        layer.qkv.weight.zero_()
        output = layer(torch.zeros(1, 4, 16, dtype=torch.float16))
    assert torch.isfinite(output).all()


def test_delta_rule_predicts_from_decayed_state():
    state = torch.eye(2).reshape(1, 1, 2, 2)
    key = query = torch.tensor([[[1.0, 0.0]]])
    value = torch.zeros(1, 1, 2)
    alpha = torch.tensor([[[[0.5], [1.0]]]])
    beta = torch.ones(1, 1, 1)
    next_state, output = delta_rule_step(state, key, value, query, alpha, beta)
    torch.testing.assert_close(next_state, torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]]))
    torch.testing.assert_close(output, torch.zeros(1, 1, 2))


def test_causal_prefix_is_unchanged_by_future_tokens():
    torch.manual_seed(0)
    model = MicroK3(tiny_config()).eval()
    a, b = torch.randint(0, 32, (2, 8))
    b[:4] = a[:4]
    with torch.no_grad():
        ya, _ = model(a[None])
        yb, _ = model(b[None])
    torch.testing.assert_close(ya[:, :4], yb[:, :4], atol=2e-5, rtol=2e-5)


def test_depth_mixing_starts_uniform_over_sources():
    """Zero depth queries mean AttnRes begins as a plain average of every source."""
    sources = [torch.randn(2, 3, 4) for _ in range(3)]
    mixed = MicroK3._mix_depth(torch.zeros(4), sources)
    torch.testing.assert_close(mixed, sum(sources) / len(sources))


def test_cached_decoding_matches_full_forward():
    torch.manual_seed(0)
    model = MicroK3(tiny_config()).eval()
    tokens = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        full, _ = model(tokens)
        caches = model.caches()
        pieces = [model(chunk, caches=caches)[0] for chunk in tokens.split((5, 2, 1), dim=1)]
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="caches"):
        model(tokens, caches=[LayerCache()])


def test_quantile_balancing_updates_next_step_bias_only_during_training():
    moe = StableLatentMoE(tiny_config()).train()
    with torch.no_grad():
        slopes = torch.linspace(-0.2, 0.2, 4)[:, None]
        moe.router.weight.copy_(slopes.expand_as(moe.router.weight))
    x = torch.ones(2, 8, 16)
    before = moe.router_bias.clone()
    moe(x)
    after = moe.router_bias.clone()
    assert not torch.equal(before, after)
    torch.testing.assert_close(after.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)

    moe.eval()
    moe(x)
    torch.testing.assert_close(moe.router_bias, after)


def test_router_bias_steers_dispatch_but_never_mixture_weights():
    """The whole point of aux-loss-free balancing: bias picks experts, scores weight them."""
    torch.manual_seed(0)
    moe = StableLatentMoE(tiny_config()).eval()
    x = torch.randn(2, 4, 16)
    baseline = moe(x)

    with torch.no_grad():
        moe.router_bias.add_(3.0)
    torch.testing.assert_close(moe(x), baseline)

    starved = int(moe.last_load.argmin())
    with torch.no_grad():
        moe.router_bias.zero_()
        moe.router_bias[starved] = 10.0
    routed = moe(x)
    assert moe.last_load[starved] == x.shape[0] * x.shape[1]
    assert not torch.allclose(routed, baseline)


def test_final_layer_is_always_global_attention():
    model = MicroK3(tiny_config(layers=5))
    assert isinstance(model.blocks[3].mix, GatedAttention)
    assert isinstance(model.blocks[4].mix, GatedAttention)


def test_dense_layers_use_plain_ffn_before_moe():
    model = MicroK3(tiny_config(dense_layers=2))
    kinds = [type(block.ffn) for block in model.blocks]
    assert kinds == [SiTUGLU, SiTUGLU, StableLatentMoE, StableLatentMoE]


@pytest.mark.parametrize(
    "overrides",
    [
        {"layers": 0},
        {"heads": 0},
        {"dim": 15},
        {"top_k": 0},
        {"top_k": 5},
        {"dense_layers": 5},
        {"dropout": 1.0},
        {"vision_layers": -1},
        {"vision_layers": 1, "vision_dim": 15},
        {"vision_layers": 1, "vision_dim": 8, "vision_heads": 4},
        {"vision_layers": 1, "patch_size": 0},
        {"mx_block": 31},
        {"mx_block": 0},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides)


def test_mx_grid_values_survive_quantization_exactly():
    """Anything already on the E2M1 grid, at any power-of-two block scale, round-trips."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for scale in (2.0**-6, 1.0, 2.0**9):
        block = torch.cat((grid, -grid)) * scale
        torch.testing.assert_close(quantize_mx(block, MXFP4, block=16), block)


def test_mxfp4_rounds_halfway_magnitudes_to_even_codes():
    """The tie rule: a midpoint lands on the even code, which is not rounding away from zero."""
    block = torch.tensor([[6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]])
    expected = torch.tensor([[6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]])
    torch.testing.assert_close(quantize_mx(block, MXFP4, block=8), expected)


def test_mx_shared_scale_is_one_power_of_two_per_block():
    blocks = torch.tensor([[[1.0, 0.5], [1000.0, 0.001]]])
    scale = mx_shared_scale(blocks, MXFP4)
    torch.testing.assert_close(torch.log2(scale).flatten(), torch.tensor([-2.0, 7.0]))
    # A loud block cannot spend the quiet block's precision: 0.5 stays exact next door.
    torch.testing.assert_close(quantize_mx(blocks.flatten(), MXFP4, block=2)[:2], blocks.flatten()[:2])


def test_mxfp8_elements_keep_more_precision_than_mxfp4():
    torch.manual_seed(0)
    x = torch.randn(64)
    coarse = (quantize_mx(x, MXFP4, 32) - x).abs().mean()
    fine = (quantize_mx(x, MXFP8, 32) - x).abs().mean()
    assert fine < coarse / 4


def test_mx_padding_matches_one_short_block():
    """A width the block size does not divide pads with zeros, which cannot shift a maximum."""
    torch.manual_seed(0)
    x = torch.randn(3, 12)
    torch.testing.assert_close(quantize_mx(x, MXFP4, 32), quantize_mx(x, MXFP4, 12))


def test_mxfp4_packs_two_codes_per_byte_and_unpacks_exactly():
    torch.manual_seed(0)
    x = torch.randn(5, 64)
    codes, exponents = pack_mxfp4(x, 32)
    assert codes.dtype == exponents.dtype == torch.uint8
    assert codes.shape == (5, 2, 16) and exponents.shape == (5, 2)
    torch.testing.assert_close(unpack_mxfp4(codes, exponents, width=64), quantize_mx(x, MXFP4, 32))


def test_straight_through_estimator_hands_back_the_incoming_gradient():
    x = torch.tensor([0.31, 1.42, -2.7], requires_grad=True)
    quantized = fake_quantize_mx(x, MXFP4, block=4)
    torch.testing.assert_close(quantized.detach(), quantize_mx(x.detach(), MXFP4, 4))
    quantized.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(3))


def test_quantization_reaches_routed_experts_and_nothing_else():
    """The report keeps routers, latent projections, shared experts, and attention in higher precision."""
    model = MicroK3(tiny_config(mx_qat=True))
    quantized = [name for name, module in model.named_modules() if getattr(module, "mx_block", 0)]
    assert quantized == [f"blocks.{layer}.ffn.experts.{index}" for layer in (1, 2, 3) for index in range(4)]
    assert model.blocks[0].ffn.mx_block == 0  # the dense first layer is not a routed expert
    assert model.blocks[1].ffn.shared.mx_block == 0


def test_quantization_moves_the_routed_output_but_not_the_shared_path():
    torch.manual_seed(0)
    plain = StableLatentMoE(tiny_config()).eval()
    quantized = StableLatentMoE(tiny_config(mx_qat=True)).eval()
    quantized.load_state_dict(plain.state_dict())
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        assert not torch.allclose(plain(x), quantized(x))
        torch.testing.assert_close(plain.shared(x), quantized.shared(x))


def test_quantized_model_overfits_one_batch():
    """Four-bit expert weights still learn, because the estimator keeps the gradient path real."""
    torch.manual_seed(0)
    model = MicroK3(tiny_config(mx_qat=True))
    x = torch.randint(0, 32, (2, 8))
    y = torch.roll(x, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def test_mxfp4_footprint_counts_routed_experts_at_four_and_a_quarter_bits():
    model = MicroK3(Config(mx_qat=True))
    cfg = model.cfg
    weights, stored = mxfp4_footprint(model)
    routed = (cfg.layers - cfg.dense_layers) * cfg.experts * 3 * cfg.latent_dim * cfg.expert_hidden
    assert weights == routed
    assert stored == routed // 2 + routed // cfg.mx_block  # 4 bits per weight, 8 per block scale
    assert mxfp4_footprint(MicroK3(Config())) == (0, 0)


def test_packed_tower_matches_encoding_each_image_alone():
    """Mixed resolutions travel as one sequence; the block-diagonal mask keeps them apart."""
    torch.manual_seed(0)
    model = MicroK3(tiny_vision_config()).eval()
    images = [torch.rand(1, 16, 16), torch.rand(1, 24, 8)]
    with torch.no_grad():
        packed = model.encode_images(images)
        alone = [model.encode_images([image])[0] for image in images]
    assert [tuple(tokens.shape) for tokens in packed] == [(4, 16), (3, 16)]
    for together, apart in zip(packed, alone, strict=True):
        torch.testing.assert_close(together, apart, atol=1e-5, rtol=1e-5)


def test_patchify_and_pixel_shuffle_keep_the_picture_in_order():
    """Each patch holds its own pixels, and one shuffled token holds a 2x2 patch neighbourhood."""
    model = MicroK3(tiny_vision_config(dim=64))
    tower = model.vision
    with torch.no_grad():  # identity weights, so tokens are literally the pixels they cover
        tower.embed.weight.copy_(torch.eye(16))
        tower.project.weight.copy_(torch.eye(64))
    patches, rows, columns = tower.patchify(torch.arange(64.0).reshape(1, 8, 8))
    assert (rows, columns) == (2, 2)
    assert patches[0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    assert patches[1, 0] == 4  # patch (0, 1) starts four pixels along the row
    assert patches[2, 0] == 32  # patch (1, 0) starts four pixels down
    merged = tower.shuffle(patches, rows, columns)
    assert merged.shape == (1, 64)
    torch.testing.assert_close(merged[0], patches.flatten())


def test_rope_2d_logits_depend_only_on_the_patch_offset():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def logit(rows, columns):
        cos, sin = rope_2d(torch.tensor(rows).float(), torch.tensor(columns).float(), head_dim=8)
        return (apply_rope(q, cos[:1], sin[:1]) * apply_rope(k, cos[1:], sin[1:])).sum()

    torch.testing.assert_close(logit((0, 1), (0, 2)), logit((3, 4), (5, 7)))
    assert not torch.isclose(logit((0, 1), (0, 2)), logit((0, 2), (0, 1)))


def test_prefix_targets_shift_captions_and_ignore_padding():
    """The last image token predicts the caption's first byte; padding predicts nothing."""
    tokens = torch.tensor([[ord("h"), ord("i"), ord("\n"), ord("\n")]])
    targets = prefix_targets(tokens, torch.tensor([3]), prefix_len=4)
    assert targets.tolist() == [[-100, -100, -100, ord("h"), ord("i"), ord("\n"), -100, -100]]
    with pytest.raises(ValueError, match="at least 1"):
        prefix_targets(tokens, torch.tensor([3]), prefix_len=0)


def test_image_prefix_extends_the_sequence_and_is_validated():
    torch.manual_seed(0)
    model = MicroK3(tiny_vision_config()).eval()
    prefix = torch.stack(model.encode_images(torch.rand(2, 1, 16, 16)))
    tokens = torch.randint(0, 32, (2, 3))
    logits, loss = model(tokens, prefix_targets(tokens, torch.tensor([3, 2]), prefix.shape[1]), prefix=prefix)
    assert logits.shape == (2, 4 + 3, 256)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="prefix must have shape"):
        model(tokens, prefix=prefix[:1])
    with pytest.raises(ValueError, match=r"prefix \+ time"):
        model(tokens, tokens, prefix=prefix)
    empty = torch.empty(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty sequence"):
        model.generate(empty, count=0, prefix=prefix[:, :0])
    with pytest.raises(ValueError, match="prefix must have shape"):
        model.generate(empty, count=0, prefix=prefix[:1])
    torch.testing.assert_close(model.generate(empty, count=0, prefix=prefix), empty)


def test_cached_decoding_with_an_image_prefix_matches_a_full_forward():
    """The picture is encoded once, into the prefill; the caches carry it from there."""
    torch.manual_seed(0)
    model = MicroK3(tiny_vision_config()).eval()
    prefix = torch.stack(model.encode_images(torch.rand(1, 1, 16, 16)))
    tokens = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        full, _ = model(tokens, prefix=prefix)
        caches = model.caches()
        prefill, _ = model(tokens[:, :4], caches=caches, prefix=prefix)
        rest, _ = model(tokens[:, 4:], caches=caches)
    torch.testing.assert_close(torch.cat((prefill, rest), dim=1), full, atol=1e-5, rtol=1e-5)


def test_captions_come_from_the_picture_with_an_empty_prompt():
    torch.manual_seed(0)
    model = MicroK3(tiny_vision_config()).eval()
    images, _, _ = shapes_batch(2, 16)
    read = caption(model, images, count=4, temperature=0.8)
    assert len(read) == 2
    assert all(isinstance(text, str) for text in read)


def test_vision_prefix_trains_the_tower():
    torch.manual_seed(0)
    model = MicroK3(tiny_vision_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    images, tokens, lengths = shapes_batch(4, 16)
    losses = []
    for _ in range(10):
        prefix = torch.stack(model.encode_images(images))
        _, loss = model(tokens, prefix_targets(tokens, lengths, prefix.shape[1]), prefix=prefix)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert torch.isfinite(model.vision.embed.weight.grad).all()
    assert losses[-1] < losses[0] - 0.5


def test_vision_qkv_is_grouped_by_the_towers_own_head_count():
    model = MicroK3(tiny_vision_config(vision_dim=32, vision_heads=4))
    matrices, _ = parameter_groups(model)
    assert id(model.vision.blocks[0].qkv.weight) in {id(p) for p in matrices[12]}
    assert id(model.blocks[0].mix.qkv.weight) in {id(p) for p in matrices[6]}


def test_shapes_batch_pairs_pictures_with_their_captions():
    torch.manual_seed(0)
    images, tokens, lengths = shapes_batch(6, 24)
    assert images.shape == (6, 1, 24, 24)
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert tokens.shape[1] == max(map(len, SHAPES)) + 1
    for row, length in zip(tokens.tolist(), lengths.tolist(), strict=True):
        assert bytes(row[:length]).decode() in [f"{name}\n" for name in SHAPES]
    with pytest.raises(ValueError, match="positive"):
        shapes_batch(0, 24)


def test_tower_rejects_images_it_cannot_patchify():
    model = MicroK3(tiny_vision_config())
    with pytest.raises(ValueError, match=r"multiples of 2 \* patch_size"):
        model.encode_images([torch.rand(1, 16, 12)])
    with pytest.raises(ValueError, match="channels"):
        model.encode_images([torch.rand(3, 16, 16)])
    with pytest.raises(ValueError, match="floating"):
        model.encode_images([torch.randint(0, 2, (1, 16, 16))])
    with pytest.raises(ValueError, match="no vision tower"):
        MicroK3(tiny_config()).encode_images([torch.rand(1, 16, 16)])


def test_sample_batch_includes_final_valid_window(monkeypatch):
    observed = {}

    def choose_last(low, high, size, *, device):
        observed["bounds"] = (low, high)
        return torch.full(size, high - 1, device=device)

    monkeypatch.setattr(torch, "randint", choose_last)
    data = torch.arange(10)
    x, y = sample_batch(data, block_size=8, batch_size=2)
    assert observed["bounds"] == (0, 2)
    torch.testing.assert_close(x[0], data[1:9])
    torch.testing.assert_close(y[0], data[2:10])


def test_sample_batch_explains_short_corpus():
    with pytest.raises(ValueError, match="at least 9"):
        sample_batch(torch.arange(8), block_size=8, batch_size=1)


def test_generation_runs_past_block_size():
    torch.manual_seed(0)
    model = MicroK3(tiny_config()).eval()
    prompt = torch.randint(0, 32, (1, 8))
    output = model.generate(prompt, count=5)
    assert output.shape == (1, 13)


def test_generation_validates_inputs_and_preserves_prompt():
    torch.manual_seed(0)
    model = MicroK3(tiny_config()).eval()
    prompt = torch.tensor([[1, 2]])
    output = model.generate(prompt, count=3, temperature=0.8)
    assert output.shape == (1, 5)
    torch.testing.assert_close(output[:, :2], prompt)
    torch.testing.assert_close(model.generate(prompt, count=0), prompt)

    with pytest.raises(ValueError, match="non-negative"):
        model.generate(prompt, count=-1)
    with pytest.raises(ValueError, match="positive"):
        model.generate(prompt, count=1, temperature=0)
    invalid_shapes = (
        torch.tensor([1]),
        torch.empty(0, 1, dtype=torch.long),
        torch.empty(1, 0, dtype=torch.long),
    )
    for invalid in invalid_shapes:
        with pytest.raises(ValueError, match="non-empty"):
            model.generate(invalid, count=0)
    with pytest.raises(ValueError, match="integer dtype"):
        model.generate(torch.ones(1, 1), count=1)
    with pytest.raises(ValueError, match=r"\[0, 32\)"):
        model.generate(torch.tensor([[32]]), count=1)


@pytest.mark.parametrize("device", ["bogus", "xpu", "meta"])
def test_unsupported_device_is_an_argument_error_not_a_traceback(device):
    with pytest.raises(argparse.ArgumentTypeError):
        available_device(device)


def test_terminal_text_escapes_control_sequences():
    payload = b"sample\n\t\x1b]0;changed\x07\x7f\xc2\x85"
    assert terminal_text(payload) == "sample\n\t\\x1b]0;changed\\x07\\x7f\\x85"


def cli(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["microk3", "--device", "cpu", *arguments])
    main()


def test_cli_trains_and_samples_from_a_corpus(tmp_path, monkeypatch, capsys):
    data = tmp_path / "tiny.txt"
    data.write_bytes(b"microK3 predicts the next byte. " * 8)
    cli(
        monkeypatch,
        "--data",
        str(data),
        "--steps",
        "2",
        "--block-size",
        "16",
        "--batch-size",
        "2",
        "--generate",
        "8",
        "--optimizer",
        "muon",
    )
    printed = capsys.readouterr().out
    assert "parameters" in printed
    assert "step 0001" in printed


def test_cli_trains_the_tower_and_reads_pictures(monkeypatch, capsys):
    cli(monkeypatch, "--vision", "--steps", "2", "--batch-size", "2", "--generate", "8")
    printed = capsys.readouterr().out
    assert "2-layer tower" in printed
    assert "pictures read" in printed


def test_cli_reports_the_mxfp4_footprint(monkeypatch, capsys):
    cli(
        monkeypatch,
        "--quantize",
        "--steps",
        "1",
        "--block-size",
        "16",
        "--batch-size",
        "2",
        "--generate",
        "0",
    )
    assert "4.25 bits per routed weight" in capsys.readouterr().out


def test_cli_explains_a_corpus_smaller_than_one_window(tmp_path, monkeypatch):
    data = tmp_path / "small.txt"
    data.write_bytes(b"too short")
    with pytest.raises(SystemExit):
        cli(monkeypatch, "--data", str(data), "--steps", "1", "--block-size", "16")
