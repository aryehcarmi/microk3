import torch

from microk3 import Config, MicroK3, SiTUGLU


def tiny_config():
    return Config(vocab_size=32, dim=16, latent_dim=8, expert_hidden=12,
                  layers=4, heads=2, experts=4, top_k=2, block_size=8)


def test_forward_backward_and_routing():
    model = MicroK3(tiny_config())
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    loss.backward()
    for block in model.blocks:
        assert block.moe.last_load.sum() == x.numel() * 2


def test_situ_glu_is_finite_for_extreme_inputs():
    layer = SiTUGLU(4, 8)
    assert torch.isfinite(layer(torch.full((2, 4), 1e6))).all()


def test_causal_prefix_is_unchanged_by_future_tokens():
    model = MicroK3(tiny_config()).eval()
    a, b = torch.randint(0, 32, (2, 8))
    b[:4] = a[:4]
    with torch.no_grad():
        ya, _ = model(a[None])
        yb, _ = model(b[None])
    torch.testing.assert_close(ya[:, :4], yb[:, :4], atol=2e-5, rtol=2e-5)
