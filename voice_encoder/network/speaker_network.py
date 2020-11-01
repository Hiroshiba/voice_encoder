import torch.nn as nn
from torch import Tensor


class SpeakerNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size, output_size),
        )

    def forward(self, x: Tensor):
        return self.layers(x)
