import torch
from torch import Tensor
from torch.nn import Linear
from torch.jit import ScriptModule, script_method

# helper functions

def exists(v):
    return v is not None

# main class

class LightRecurrentUnitCell(ScriptModule):
    def __init__(self, dim):
        super().__init__()
        self.weights_input = Linear(dim, dim * 2, bias = False)
        self.hidden_forget = Linear(dim, dim)

    @script_method
    def forward(
        self,
        x: Tensor,
        hidden: Tensor
    ):

        # derive the next hidden as well as the forget gate contribution from the input

        next_input, input_forget = self.weights_input(x).chunk(2, dim = -1)

        next_input = torch.tanh(next_input)

        # get the forget gate contribution from previous hidden

        hidden_forget = self.hidden_forget(hidden)

        # calculate forget gate

        forget_gate = (hidden_forget + input_forget).sigmoid()

        # next hidden = hidden * (1. - forget_gate) + next_hidden * forget_gate

        next_hidden = hidden.lerp(next_input, forget_gate)

        return next_hidden
