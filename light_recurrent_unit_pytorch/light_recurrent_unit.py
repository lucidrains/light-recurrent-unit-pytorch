from __future__ import annotations

import einx
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList
from torch_einops_utils.shape import shape, size

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# a single LRU cell

class LightRecurrentUnitCell(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        *,
        heads = 1,
        proj_input = True,
        learned_init_hidden = False
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)

        assert heads >= 1
        assert proj_input or heads == 1, 'heads > 1 requires the input projection'

        self.heads = heads
        self.dim_hidden = dim_hidden
        self.is_multi_head = heads > 1

        self.to_next_hidden = nn.Sequential(Linear(dim, heads * dim_hidden, bias = False), nn.Tanh()) if proj_input else nn.Identity()

        self.to_input_forget = Linear(dim, dim_hidden, bias = False)
        self.to_hidden_forget = Linear(dim_hidden, dim_hidden)

        # multi head - softmax selection over the candidates

        self.to_attn = Linear(heads * dim_hidden, heads, bias = False) if self.is_multi_head else None

        # initial hidden

        init_hidden = torch.zeros(dim_hidden)

        if learned_init_hidden:
            self.init_hidden = nn.Parameter(init_hidden)
        else:
            self.register_buffer('init_hidden', init_hidden)

    def forward(
        self,
        x: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        next_hidden, input_forget = self.project_input(x)
        return self.forward_step(next_hidden, input_forget, hidden)

    def project_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # derive the next hidden as well as the forget gate contribution from the input

        next_hidden, input_forget = self.to_next_hidden(x), self.to_input_forget(x)

        # multi head - softmax select over the candidates

        if self.is_multi_head:
            attn = self.to_attn(next_hidden).softmax(dim = -1)
            weighted = einx.multiply('... k, ... (k d) -> ... k d', attn, next_hidden, k = self.heads)
            next_hidden = einx.sum('... k d -> ... d', weighted)

        return next_hidden, input_forget

    def forward_step(
        self,
        next_hidden: Tensor,
        input_forget: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        if not exists(hidden):
            hidden = self.init_hidden

        # get the forget gate contribution from previous hidden

        hidden_forget = self.to_hidden_forget(hidden)

        # calculate forget gate

        forget_gate = (hidden_forget + input_forget).sigmoid()

        # next hidden = hidden * (1. - forget_gate) + next_hidden * forget_gate

        return hidden.lerp(next_hidden, forget_gate)

# LRU layer

class LightRecurrentUnitLayer(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        *,
        heads = 1,
        proj_input = True,
        learned_init_hidden = False
    ):
        super().__init__()
        self.cell = LightRecurrentUnitCell(dim, dim_hidden, heads = heads, proj_input = proj_input, learned_init_hidden = learned_init_hidden)

    def forward(
        self,
        x: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        # batch first always (batch, time, dim)

        batch, time = shape(x, 'b t _ -> b t')

        # derive all input dependent quantities for the entire sequence at once

        next_hiddens, input_forgets = self.cell.project_input(x)

        # recurrent roll out, only hidden dependent quantities remain in the loop

        dim_hidden = size(next_hiddens, 'b t d -> d')
        out = x.new_empty(batch, time, dim_hidden)

        for step, (next_hidden, input_forget) in enumerate(zip(next_hiddens.unbind(dim = 1), input_forgets.unbind(dim = 1))):
            hidden = self.cell.forward_step(next_hidden, input_forget, hidden)
            out[:, step] = hidden

        return out

# Stacked LRU

class LightRecurrentUnit(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 1,
        proj_input: bool | tuple[bool, ...] = True,
        learned_init_hidden = False
    ):
        super().__init__()

        if not isinstance(proj_input, tuple):
            proj_input = (proj_input,) * depth

        assert len(proj_input) == depth

        self.layers = ModuleList([LightRecurrentUnitLayer(dim, heads = heads, proj_input = layer_proj_input, learned_init_hidden = learned_init_hidden) for layer_proj_input in proj_input])

    def forward(
        self,
        x: Tensor,
        hiddens: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:

        next_hiddens = []

        for i, layer in enumerate(self.layers):

            layer_hiddens = None

            if exists(hiddens):
                layer_hiddens = hiddens[i]

            x = layer(x, layer_hiddens)

            next_hiddens.append(x[:, -1])

        return x, next_hiddens

# an improvised variant where stacked LRU has residual at each layer but gated with an LRU itself

class GatedLightRecurrentUnit(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 1,
        learned_init_hidden = False,
        num_layers_per_depth = 2
    ):
        super().__init__()
        self.gate = LightRecurrentUnitCell(dim, heads = heads)

        layers = []
        for _ in range(depth):
            layer = LightRecurrentUnit(dim, depth = num_layers_per_depth, heads = heads, learned_init_hidden = learned_init_hidden)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        hiddens: list[list[Tensor]] | None = None
    ) -> tuple[Tensor, list[list[Tensor]]]:

        next_hiddens = []

        for i, layer in enumerate(self.layers):

            layer_hiddens = None

            if exists(hiddens):
                layer_hiddens = hiddens[i]

            layer_out, layer_hiddens = layer(x, layer_hiddens)

            next_hiddens.append(layer_hiddens)

            x = self.gate(layer_out, x)

        return x, next_hiddens

# LRU Block

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1.)

class LightRecurrentUnitBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 1,
        learned_init_hidden = True,
        depth_gated_lru = True,
        has_ff_block = False,
        ff_expansion_factor = 4,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        lru_klass = GatedLightRecurrentUnit if depth_gated_lru else LightRecurrentUnit

        self.lru = lru_klass(dim = dim, depth = depth, heads = heads, learned_init_hidden = learned_init_hidden)

        self.has_ff_block = has_ff_block

        if not has_ff_block:
            return

        dim_ff_inner = int(dim * ff_expansion_factor)

        self.ff = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_ff_inner),
            nn.GELU(),
            Linear(dim_ff_inner, dim)
        )

    def forward(self, x):
        lru_out, _ = self.lru(self.norm(x))

        x = lru_out + x

        if not self.has_ff_block:
            return x

        return self.ff(x) + x
