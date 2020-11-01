import torch
import torch.nn as nn
from torch import Tensor


class F0Network(nn.Module):
    def __init__(self, input_size: int, speaker_size: int, speaker_embedding_size: int):
        super().__init__()

        self.linear = nn.Linear(input_size + speaker_embedding_size, 1)
        self.speaker_embedder = nn.Embedding(
            num_embeddings=speaker_size,
            embedding_dim=speaker_embedding_size,
        )

    def forward(self, x: Tensor, speaker: Tensor):
        speaker = self.speaker_embedder(speaker)
        speaker = speaker.unsqueeze(1)
        speaker = speaker.expand(speaker.shape[0], x.shape[1], speaker.shape[2])
        h = torch.cat((x, speaker), dim=2)
        return self.linear(h).squeeze()
