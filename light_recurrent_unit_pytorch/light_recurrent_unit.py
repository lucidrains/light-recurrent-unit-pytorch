from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Module, ModuleList
from torch.jit import ScriptModule, script_method

# a single LRU cell

class LightRecurrentUnitCell(ScriptModule):
    def __init__(self, dim):
        super().__init__()
        self.input_proj = Linear(dim, dim * 2, bias = False)
        self.hidden_proj = Linear(dim, dim)

    @script_method
    def forward(
        self,
        x: Tensor,
        hidden: Tensor | None = None
    ):

        if hidden is None:
            hidden = torch.zeros_like(x)

        # derive the next hidden as well as the forget gate contribution from the input

        next_hidden, input_forget = self.input_proj(x).chunk(2, dim = -1)

        # get the forget gate contribution from previous hidden

        hidden_forget = self.hidden_proj(hidden)

        # calculate forget gate

        forget_gate = (hidden_forget + input_forget).sigmoid()

        # next hidden = hidden * (1. - forget_gate) + next_hidden * forget_gate

        next_hidden = hidden.lerp(next_hidden.tanh(), forget_gate)

        return next_hidden

# LRU layer

class LightRecurrentUnitLayer(ScriptModule):
    def __init__(self, dim):
        super().__init__()
        self.cell = LightRecurrentUnitCell(dim)

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
        depth = 1
    ):
        super().__init__()
        self.layers = ModuleList([LightRecurrentUnitLayer(dim) for _ in range(depth)])

    @script_method
    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers:
            x = layer(x)

        return x

# LRU Block

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1.)

class LightRecurrentUnitBlock(Module):
    def __init__(
        self,
        dim,
        depth = 1
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lru = LightRecurrentUnit(dim = dim, depth = depth)

    def forward(self, x):
        return self.lru(self.norm(x)) + x
