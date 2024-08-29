from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
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

        next_input, input_forget = self.input_proj(x).chunk(2, dim = -1)

        next_input = torch.tanh(next_input)

        # get the forget gate contribution from previous hidden

        hidden_forget = self.hidden_proj(hidden)

        # calculate forget gate

        forget_gate = (hidden_forget + input_forget).sigmoid()

        # next hidden = hidden * (1. - forget_gate) + next_hidden * forget_gate

        next_hidden = hidden.lerp(next_input, forget_gate)

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
