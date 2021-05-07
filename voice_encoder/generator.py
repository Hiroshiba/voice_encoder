from pathlib import Path
from typing import Optional, Union

import numpy
import torch
import torch.nn.functional as F
from torch import nn

from voice_encoder.config import Config
from voice_encoder.network.f0_network import F0Network, create_f0_network
from voice_encoder.network.network import create_phoneme_network, create_voiced_network
from voice_encoder.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        voiced_network: Optional[Union[nn.Linear, Path]],
        f0_network: Optional[Union[F0Network, Path]],
        phoneme_network: Optional[Union[nn.Module, Path]],
        use_gpu: bool,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.scale = int(numpy.prod(config.network.scale_list))

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        if voiced_network is not None:
            if isinstance(voiced_network, Path):
                state_dict = torch.load(voiced_network, map_location=self.device)
                voiced_network = create_voiced_network(config.network)
                voiced_network.load_state_dict(state_dict)
            voiced_network = voiced_network.eval().to(self.device)
        self.voiced_network = voiced_network

        if f0_network is not None:
            if isinstance(f0_network, Path):
                state_dict = torch.load(f0_network, map_location=self.device)
                f0_network = create_f0_network(config.network)
                f0_network.load_state_dict(state_dict)
            f0_network = f0_network.eval().to(self.device)
        self.f0_network = f0_network

        if phoneme_network is not None:
            if isinstance(phoneme_network, Path):
                state_dict = torch.load(phoneme_network, map_location=self.device)
                phoneme_network = create_phoneme_network(config.network)
                phoneme_network.load_state_dict(state_dict)
            phoneme_network = phoneme_network.eval().to(self.device)
        self.phoneme_network = phoneme_network

    def generate(
        self,
        wave: Union[numpy.ndarray, torch.Tensor],
        to_voiced_scaler: bool = False,
        to_f0_scaler: bool = False,
        to_phoneme_onehot: bool = False,
    ):
        if isinstance(wave, numpy.ndarray):
            wave = torch.from_numpy(wave)
        wave = wave.to(self.device)

        batch_size = wave.shape[0]
        length = wave.shape[1] // self.scale

        with torch.no_grad():
            if not (to_voiced_scaler or to_f0_scaler or to_phoneme_onehot):
                output = self.predictor(wave)
            else:
                features = self.predictor(wave, return_with_splited=True)
                voiced_feature = features["voiced"]
                f0_feature = features["f0"]
                phoneme_feature = features["phoneme"]

                if to_voiced_scaler:
                    voiced_feature = voiced_feature.transpose(1, 2).reshape(
                        batch_size * length, -1
                    )
                    voiced_output = self.voiced_network(voiced_feature)
                    voiced_feature = (
                        voiced_output.argmax(1)
                        .to(voiced_feature.dtype)
                        .reshape(batch_size, length)
                        .unsqueeze(1)
                    )

                if to_f0_scaler:
                    f0_feature = f0_feature.transpose(1, 2)
                    f0_feature = self.f0_network(x=f0_feature).unsqueeze(1)

                if to_phoneme_onehot:
                    phoneme_feature = phoneme_feature.transpose(1, 2).reshape(
                        batch_size * length, -1
                    )
                    phoneme_output = self.phoneme_network(phoneme_feature)
                    phoneme_feature = (
                        F.one_hot(phoneme_output.argmax(1), phoneme_output.shape[1])
                        .to(phoneme_feature.dtype)
                        .reshape(batch_size, length, -1)
                        .transpose(1, 2)
                    )

                output = torch.cat([voiced_feature, f0_feature, phoneme_feature], dim=1)
        return output.cpu().numpy()
