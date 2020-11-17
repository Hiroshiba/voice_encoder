from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class F0Network(nn.Module):
    def __init__(
        self, input_size: int, speaker_size: int, speaker_embedding_size: Optional[int]
    ):
        super().__init__()

        if speaker_embedding_size is not None:
            self.linear = nn.Linear(input_size + speaker_embedding_size, 1)
            self.speaker_embedder = nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
        else:
            self.linear = nn.Linear(input_size, 1)
            self.speaker_embedder = None

    def forward(self, x: Tensor, speaker: Tensor):
        if self.speaker_embedder is not None:
            speaker = self.speaker_embedder(speaker)
            speaker = speaker.unsqueeze(1)
            speaker = speaker.expand(speaker.shape[0], x.shape[1], speaker.shape[2])
            x = torch.cat((x, speaker), dim=2)
        return self.linear(x).squeeze()
