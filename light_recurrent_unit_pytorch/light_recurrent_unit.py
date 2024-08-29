from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Linear
from torch.jit import ScriptModule, script_method

# helper functions

def exists(v):
    return v is not None

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
        hidden: Tensor
    ):

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
    def __init__(self, dim,):
        super().__init__()
        self.cell = LightRecurrentUnitCell(dim)

    @script_method
    def forward(
        self,
        x: Tensor,
        hidden: Tensor
    ) -> Tensor:
        # assume always (batch, time, dim)

        inputs = x.unbind(dim = 1)
        next_hiddens: list[Tensor] = []

        for timestep_input in inputs:
            next_hidden = self.cell(timestep_input, hidden)
            next_hiddens.append(next_hidden)

        return torch.stack(next_hiddens, dim = 1)
