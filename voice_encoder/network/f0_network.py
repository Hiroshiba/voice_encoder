import torch.nn as nn
from torch import Tensor
from voice_encoder.config import NetworkConfig


class F0Network(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: Tensor):
        return self.linear(x).squeeze(2)


def create_f0_network(config: NetworkConfig):
    return F0Network(input_size=config.f0_feature_size)
