from dataclasses import dataclass

from torch import nn
from voice_encoder.config import NetworkConfig
from voice_encoder.network.ada_cos import AdaCos, SubscaleAdaCos
from voice_encoder.network.f0_network import F0Network, create_f0_network
from voice_encoder.network.predictor import Predictor, create_predictor
from voice_encoder.network.speaker_network import SpeakerNetwork


@dataclass
class Networks:
    predictor: Predictor
    voiced_network: nn.Linear
    f0_network: F0Network
    phoneme_network: nn.Module
    speaker_network: nn.Module


def create_voiced_network(config: NetworkConfig):
    return nn.Linear(
        in_features=config.voiced_feature_size,
        out_features=2,
    )


def create_phoneme_network(config: NetworkConfig):
    if config.phoneme_subscale_size is None:
        return AdaCos(
            feature_size=config.phoneme_feature_size,
            class_size=config.phoneme_class_size,
        )
    else:
        return SubscaleAdaCos(
            feature_size=config.phoneme_feature_size,
            class_size=config.phoneme_class_size,
            subscale_size=config.phoneme_subscale_size,
        )


def create_network(config: NetworkConfig):
    feature_size = (
        config.voiced_feature_size
        + config.f0_feature_size
        + config.phoneme_feature_size
    )
    return Networks(
        predictor=create_predictor(config),
        voiced_network=create_voiced_network(config),
        f0_network=create_f0_network(config),
        phoneme_network=create_phoneme_network(config),
        speaker_network=SpeakerNetwork(
            input_size=feature_size,
            output_size=config.speaker_size,
        ),
    )
