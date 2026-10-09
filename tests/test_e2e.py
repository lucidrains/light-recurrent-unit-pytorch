import pytest
import torch

from light_recurrent_unit_pytorch import LightRecurrentUnit, LightRecurrentUnitBlock

@pytest.mark.parametrize('heads', (1, 2, 4))
def test_light_recurrent_unit_e2e(heads):
    lru = LightRecurrentUnit(256, depth = 4, heads = heads)

    x = torch.randn(2, 1024, 256)

    out, layer_hiddens = lru(x)

    assert out.shape == x.shape
    assert len(layer_hiddens) == 4

    out.sum().backward()

    assert all(p.grad is not None for p in lru.parameters())

@pytest.mark.parametrize('heads', (1, 2, 4))
def test_light_recurrent_unit_block_e2e(heads):
    block = LightRecurrentUnitBlock(256, depth = 2, heads = heads, has_ff_block = True)

    x = torch.randn(2, 512, 256)

    out = block(x)

    assert out.shape == x.shape

    out.sum().backward()

    assert all(p.grad is not None for p in block.parameters())
