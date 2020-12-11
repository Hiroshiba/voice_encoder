from pathlib import Path
from typing import Optional, Union

import numpy
import torch

from voice_encoder.config import Config
from voice_encoder.network.f0_network import F0Network, create_f0_network
from voice_encoder.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        f0_network: Optional[Union[F0Network, Path]],
        use_gpu: bool,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        if f0_network is not None:
            if isinstance(f0_network, Path):
                state_dict = torch.load(f0_network)
                f0_network = create_f0_network(config.network)
                f0_network.load_state_dict(state_dict)
            f0_network = f0_network.eval().to(self.device)
        self.f0_network = f0_network

    def generate(
        self,
        wave: Union[numpy.ndarray, torch.Tensor],
        to_f0_scaler: bool = False,
    ):
        if isinstance(wave, numpy.ndarray):
            wave = torch.from_numpy(wave)
        wave = wave.to(self.device)

        with torch.no_grad():
            if not to_f0_scaler:
                output = self.predictor(wave)
            else:
                features = self.predictor(wave, return_with_splited=True)
                f0_feature = features["f0"].transpose(1, 2)
                f0_feature = self.f0_network(x=f0_feature).unsqueeze(1)
                output = torch.cat(
                    [features["voiced"], f0_feature, features["phoneme"]], dim=1
                )
        return output.cpu().numpy()
