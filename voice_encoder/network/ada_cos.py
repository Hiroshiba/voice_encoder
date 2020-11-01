r"""
https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AdaCos(nn.Module):
    def __init__(self, feature_size: int, class_size: int):
        super().__init__()
        self.s = math.sqrt(2) * math.log(class_size - 1)
        self.W = Parameter(torch.FloatTensor(class_size, feature_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, label=None):
        # x = F.normalize(x)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits

        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits)
            )
            B_avg = torch.sum(B_avg) / x.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
            )
        output = self.s * logits

        return output
