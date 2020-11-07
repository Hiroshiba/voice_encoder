from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from voice_encoder.config import ModelConfig, NetworkConfig
from voice_encoder.network.ada_cos import AdaCos
from voice_encoder.network.f0_network import F0Network
from voice_encoder.network.predictor import Predictor, create_predictor
from voice_encoder.network.speaker_network import SpeakerNetwork


@dataclass
class Networks:
    predictor: Predictor
    voiced_network: nn.Linear
    f0_network: nn.Module
    phoneme_network: AdaCos
    speaker_network: nn.Module


def create_network(config: NetworkConfig):
    feature_size = (
        config.voiced_feature_size
        + config.f0_feature_size
        + config.phoneme_feature_size
    )
    return Networks(
        predictor=create_predictor(config),
        voiced_network=nn.Linear(
            in_features=config.voiced_feature_size,
            out_features=2,
        ),
        f0_network=F0Network(
            input_size=config.f0_feature_size,
            speaker_size=config.speaker_size,
            speaker_embedding_size=config.speaker_embedding_size,
        ),
        phoneme_network=AdaCos(
            feature_size=config.phoneme_feature_size,
            class_size=config.phoneme_class_size,
        ),
        speaker_network=SpeakerNetwork(
            input_size=feature_size,
            output_size=config.speaker_size,
        ),
    )


def accuracy(output: Tensor, target: Tensor):
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        return correct.float().mean()


class Model(nn.Module):
    def __init__(self, config: ModelConfig, networks: Networks):
        super().__init__()
        self.config = config
        self.predictor = networks.predictor
        self.voiced_network = networks.voiced_network
        self.f0_network = networks.f0_network
        self.phoneme_network = networks.phoneme_network
        self.speaker_network = networks.speaker_network

    def __call__(
        self,
        wave: Tensor,
        f0: Tensor,
        phoneme: Tensor,
        speaker: Optional[Tensor] = None,
    ):
        batch_size = wave.shape[0]
        length = f0.shape[1]

        voiced = f0 != 0
        long_voiced = voiced.long()

        features = self.predictor(wave, return_with_splited=True)
        feature = features["feature"].transpose(1, 2).reshape(batch_size * length, -1)
        voiced_feature = (
            features["voiced"].transpose(1, 2).reshape(batch_size * length, -1)
        )
        f0_feature = features["f0"].transpose(1, 2)
        phoneme_feature = (
            features["phoneme"].transpose(1, 2).reshape(batch_size * length, -1)
        )

        voiced_output = self.voiced_network(voiced_feature)
        phoneme_output = self.phoneme_network(phoneme_feature, phoneme)

        voiced_loss = F.cross_entropy(voiced_output, long_voiced.reshape(-1))
        phoneme_loss = F.cross_entropy(phoneme_output, phoneme.reshape(-1))

        if speaker is not None:
            expanded_speaker = speaker.unsqueeze(1).expand(batch_size, length)

            f0_output = self.f0_network(x=f0_feature, speaker=speaker)
            speaker_output = self.speaker_network(feature.detach())

            f0_loss = F.l1_loss(f0_output[voiced], f0[voiced])
            speaker_loss = F.cross_entropy(speaker_output, expanded_speaker.reshape(-1))
            speaker_accuracy = accuracy(speaker_output, expanded_speaker.reshape(-1))
        else:
            f0_loss = 0
            speaker_loss = 0
            speaker_accuracy = 0

        predictor_loss = (
            self.config.voiced_loss_weight * voiced_loss
            + self.config.f0_loss_weight * f0_loss
            + self.config.phoneme_loss_weight * phoneme_loss
        )
        loss = predictor_loss + speaker_loss

        # report
        values = dict(
            loss=loss,
            predictor_loss=predictor_loss,
            voiced_loss=voiced_loss,
            f0_loss=f0_loss,
            phoneme_loss=phoneme_loss,
            speaker_loss=speaker_loss,
            voiced_accuracy=accuracy(voiced_output, long_voiced.reshape(-1)),
            phoneme_accuracy=accuracy(phoneme_output, phoneme.reshape(-1)),
            speaker_accuracy=speaker_accuracy,
        )
        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
