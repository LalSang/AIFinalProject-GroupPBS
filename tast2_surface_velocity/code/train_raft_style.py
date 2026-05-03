"""Train a RAFT-inspired feature-correlation model.

Full RAFT estimates optical flow from paired frames and requires flow labels.
Those labels are absent here, so this script implements a small RAFT-style
correlation block between two streams: masked building appearance and local
context appearance. The output remains the available DoriaNET damage level.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from train_common import parser_with_defaults, train_and_evaluate


class SharedFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RAFTStyleCorrelationCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.encoder = SharedFeatureEncoder()
        self.update = nn.Sequential(
            nn.Conv2d(96 * 4 + 1, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x, extra):
        masked_rgb = x[:, :3]
        context_rgb = x[:, 3:]
        f_masked = self.encoder(masked_rgb)
        f_context = self.encoder(context_rgb)
        f_masked = F.normalize(f_masked, dim=1)
        f_context = F.normalize(f_context, dim=1)
        # Local correlation summarizes feature agreement at each spatial cell.
        corr = (f_masked * f_context).sum(dim=1, keepdim=True)
        features = torch.cat([f_masked, f_context, torch.abs(f_masked - f_context), f_masked * f_context, corr], dim=1)
        return self.head(self.update(features))


def main() -> None:
    parser = parser_with_defaults("Train RAFT-inspired DoriaNET model")
    args = parser.parse_args()
    train_and_evaluate("raft_style", RAFTStyleCorrelationCNN(), args, mode="raft")


if __name__ == "__main__":
    main()

