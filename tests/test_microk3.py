import math

import pytest
import torch

from microk3 import (
    Config,
    GatedAttention,
    KimiDeltaAttention,
    MicroK3,
    SiTUGLU,
    StableLatentMoE,
    delta_rule_step,
    sample_batch,
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
    for block in model.blocks:
        assert block.moe.last_load.sum() == x.numel() * 2


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


def test_final_layer_is_always_global_attention():
    model = MicroK3(tiny_config(layers=5))
    assert isinstance(model.blocks[3].mix, GatedAttention)
    assert isinstance(model.blocks[4].mix, GatedAttention)


@pytest.mark.parametrize(
    "overrides",
    [
        {"layers": 0},
        {"heads": 0},
        {"dim": 15},
        {"top_k": 0},
        {"top_k": 5},
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
