import math

from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):

    def __init__(self, layer: nn.Linear, rank: int):
        super().__init__()
        self.weight = layer.weight.requires_grad_(False)
        self.bias = layer.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(layer.weight.new_empty((rank, layer.in_features)))
        self.lora_b = nn.Parameter(layer.weight.new_zeros((layer.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base + update
