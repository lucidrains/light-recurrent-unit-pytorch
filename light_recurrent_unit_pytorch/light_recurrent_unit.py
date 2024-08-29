import torch
from torch.nn import Module, ModuleList

# helper functions

def exists(v):
    return v is not None

# main class

class LightRecurrentUnit(Module):
    def __init__(
        self,
        dim,
        *,
        dim_hidden = None
    ):
        super().__init__()

    def forward(self, x):
        return x
