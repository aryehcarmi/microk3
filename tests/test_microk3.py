import argparse
import math
import sys

import pytest
import torch

from microk3 import (
    Config,
    GatedAttention,
    KimiDeltaAttention,
    LayerCache,
    MicroK3,
    SiTUGLU,
    StableLatentMoE,
    available_device,
    build_optimizers,
    delta_rule_step,
    learning_rate_scale,
    main,
    orthogonalize,
    parameter_groups,
    sample_batch,
    terminal_text,
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


def test_parameter_groups_cover_every_parameter_exactly_once():
    model = MicroK3(tiny_config())
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
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides)


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

    with pytest.raises(ValueError, match="non-negative"):
        model.generate(prompt, count=-1)
    with pytest.raises(ValueError, match="positive"):
        model.generate(prompt, count=1, temperature=0)
    with pytest.raises(ValueError, match="non-empty"):
        model.generate(torch.empty(1, 0, dtype=torch.long), count=1)
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


def test_cli_explains_a_corpus_smaller_than_one_window(tmp_path, monkeypatch):
    data = tmp_path / "small.txt"
    data.write_bytes(b"too short")
    with pytest.raises(SystemExit):
        cli(monkeypatch, "--data", str(data), "--steps", "1", "--block-size", "16")
