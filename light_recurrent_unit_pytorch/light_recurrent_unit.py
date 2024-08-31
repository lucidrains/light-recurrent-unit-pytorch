from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList
from torch.jit import ScriptModule, script_method

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# a single LRU cell

class LightRecurrentUnitCell(ScriptModule):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        *,
        proj_input = True,
        learned_init_hidden = False
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)

        self.to_next_hidden = nn.Sequential(Linear(dim, dim_hidden, bias = False), nn.Tanh()) if proj_input else nn.Identity()

        self.to_input_forget = Linear(dim, dim_hidden, bias = False)
        self.to_hidden_forget = Linear(dim_hidden, dim_hidden)

        # initial hidden

        init_hidden = torch.zeros(dim_hidden)

        if learned_init_hidden:
            self.register_parameter(init_hidden)
        else:
            self.register_buffer('init_hidden', init_hidden)

    @script_method
    def forward(
        self,
        x: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        if hidden is None:
            hidden = self.init_hidden

        # derive the next hidden as well as the forget gate contribution from the input

        next_hidden, input_forget = self.to_next_hidden(x), self.to_input_forget(x)

        # get the forget gate contribution from previous hidden

        hidden_forget = self.to_hidden_forget(hidden)

        # calculate forget gate

        forget_gate = (hidden_forget + input_forget).sigmoid()

        # next hidden = hidden * (1. - forget_gate) + next_hidden * forget_gate

        next_hidden = hidden.lerp(next_hidden, forget_gate)

        return next_hidden

# LRU layer

class LightRecurrentUnitLayer(ScriptModule):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        *,
        proj_input = True,
        learned_init_hidden = False
    ):
        super().__init__()
        self.cell = LightRecurrentUnitCell(dim, dim_hidden, proj_input = proj_input, learned_init_hidden = learned_init_hidden)

    @script_method
    def forward(
        self,
        x: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        # batch first always (batch, time, dim)

        inputs = x.unbind(dim = 1)
        next_hiddens: list[Tensor] = []

        for timestep_input in inputs:
            hidden = self.cell(timestep_input, hidden)
            next_hiddens.append(hidden)

        return torch.stack(next_hiddens, dim = 1)

# Stacked LRU

class LightRecurrentUnit(ScriptModule):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        proj_input: bool | tuple[bool, ...] = True,
        learned_init_hidden = False
    ):
        super().__init__()

        if not isinstance(proj_input, tuple):
            proj_input = (proj_input,) * depth

        assert len(proj_input) == depth

        self.layers = ModuleList([LightRecurrentUnitLayer(dim, proj_input = layer_proj_input, learned_init_hidden = learned_init_hidden) for layer_proj_input in proj_input])

    @script_method
    def forward(
        self,
        x: Tensor,
        hiddens: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:

        next_hiddens: list[Tensor] = []

        for i, layer in enumerate(self.layers):

            layer_hiddens: Tensor | None = None

            if hiddens is not None:
                layer_hiddens = hiddens[i]

            x = layer(x, layer_hiddens)

            next_hiddens.append(x[:, -1])

        return x, next_hiddens

# an improvised variant where stacked LRU has residual at each layer but gated with an LRU itself

GatedLayerHidden = list[list[Tensor]]

class GatedLightRecurrentUnit(ScriptModule):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        learned_init_hidden = False,
        num_layers_per_depth = 2
    ):
        super().__init__()
        self.gate = LightRecurrentUnitCell(dim)

        layers = []
        for _ in range(depth):
            layer = LightRecurrentUnit(dim, depth = num_layers_per_depth, learned_init_hidden = learned_init_hidden)
            layers.append(layer)

        self.layers = ModuleList(layers)

    @script_method
    def forward(
        self,
        x: Tensor,
        hiddens: GatedLayerHidden | None = None
    ) -> tuple[Tensor, GatedLayerHidden]:

        next_hiddens: GatedLayerHidden = []

        for i, layer in enumerate(self.layers):

            layer_hiddens: list[Tensor] | None = None

            if hiddens is not None:
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
        learned_init_hidden = True,
        depth_gated_lru = True,
        has_ff_block = False,
        ff_expansion_factor = 4,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        lru_klass = GatedLightRecurrentUnit if depth_gated_lru else LightRecurrentUnit

        self.lru = lru_klass(dim = dim, depth = depth, learned_init_hidden = learned_init_hidden)

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
