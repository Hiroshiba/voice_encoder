import torch.nn.functional as F
from voice_encoder.config import NetworkConfig
from torch import Tensor, nn


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def create_predictor(config: NetworkConfig):
    return Predictor()
