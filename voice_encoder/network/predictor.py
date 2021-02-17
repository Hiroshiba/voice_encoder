from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from voice_encoder.config import NetworkConfig
from voice_encoder.network.encoder import EncoderType, create_encoder


class Mish(nn.Module):
    def forward(self, x: Tensor):
        return x * torch.tanh(F.softplus(x))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        padding_size: int,
        scale: int,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding_size,
                stride=scale,
            )
        )
        self.activator = Mish()

    def forward(self, x: Tensor):
        return self.activator(self.conv(x))


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size_list: Sequence[int],
        scale_list: Sequence[int],
        kernel_size_list: Sequence[int],
        padding_size_list: Sequence[int],
        encoder_type: Optional[EncoderType],
        encoder_hidden_size: int,
        encoder_kernel_size: int,
        encoder_layer_num: int,
        voiced_feature_size: int,
        f0_feature_size: int,
        phoneme_feature_size: int,
    ):
        super().__init__()

        self.voiced_feature_size = voiced_feature_size
        self.f0_feature_size = f0_feature_size
        self.phoneme_feature_size = phoneme_feature_size

        feature_size = voiced_feature_size + f0_feature_size + phoneme_feature_size

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    input_size=hidden_size_list[i - 1] if i > 0 else 1,
                    output_size=hidden_size_list[i],
                    kernel_size=kernel_size_list[i],
                    padding_size=padding_size_list[i],
                    scale=scale_list[i],
                )
                for i in range(len(hidden_size_list))
            ]
        )

        last_hidden_size = hidden_size_list[-1]
        if encoder_type is None:
            self.encoder = None
        else:
            self.encoder = create_encoder(
                type=encoder_type,
                input_size=last_hidden_size,
                hidden_size=encoder_hidden_size,
                kernel_size=encoder_kernel_size,
                layer_num=encoder_layer_num,
            )
            last_hidden_size = self.encoder.output_hidden_size

        self.conv = nn.Conv1d(
            in_channels=last_hidden_size,
            out_channels=feature_size,
            kernel_size=1,
        )

    def forward(self, x: Tensor, return_with_splited: bool = False):
        h = x.unsqueeze(1)

        h = self.blocks(h)
        if self.encoder is not None:
            h = self.encoder(h)
        feature = self.conv(h)

        voiced_feature, f0_feature, phoneme_feature = torch.split(
            feature,
            [
                self.voiced_feature_size,
                self.f0_feature_size,
                self.phoneme_feature_size,
            ],
            dim=1,
        )

        voiced_feature = F.normalize(voiced_feature)
        f0_feature = F.normalize(f0_feature)
        phoneme_feature = F.normalize(phoneme_feature)

        feature = torch.cat([voiced_feature, f0_feature, phoneme_feature], dim=1)

        if return_with_splited:
            return dict(
                feature=feature,
                voiced=voiced_feature,
                f0=f0_feature,
                phoneme=phoneme_feature,
            )
        else:
            return feature


def create_predictor(config: NetworkConfig):
    return Predictor(
        encoder_type=(
            None if config.encoder_type is None else EncoderType(config.encoder_type)
        ),
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_kernel_size=config.encoder_kernel_size,
        encoder_layer_num=config.encoder_layer_num,
        hidden_size_list=config.hidden_size_list,
        scale_list=config.scale_list,
        kernel_size_list=config.kernel_size_list,
        padding_size_list=config.padding_size_list,
        voiced_feature_size=config.voiced_feature_size,
        f0_feature_size=config.f0_feature_size,
        phoneme_feature_size=config.phoneme_feature_size,
    )
