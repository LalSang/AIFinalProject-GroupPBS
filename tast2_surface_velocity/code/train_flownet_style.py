"""Train a lightweight FlowNet-style CNN baseline.

The available dataset is not optical-flow/velocity data. This baseline borrows
the FlowNet idea of a strided encoder over stacked visual channels; the input is
RGB building crop + binary building mask and the target is ordinal damage level.
"""

from __future__ import annotations

from torch import nn

from train_common import parser_with_defaults, train_and_evaluate


class FlowNetStyleCNN(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x, extra):
        return self.head(self.encoder(x))


def main() -> None:
    parser = parser_with_defaults("Train FlowNet-style DoriaNET baseline")
    args = parser.parse_args()
    train_and_evaluate("flownet_style", FlowNetStyleCNN(), args, mode="flownet")


if __name__ == "__main__":
    main()

