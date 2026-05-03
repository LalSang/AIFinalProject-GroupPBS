"""Train a segmentation-assisted CNN regression/classification baseline.

The script name follows the assignment request. Because the raw DoriaNET data
does not contain surface velocity labels, the model predicts the available
ordinal damage level using frame crops, masks, and simple mask geometry.
"""

from __future__ import annotations

import torch
from torch import nn

from train_common import parser_with_defaults, train_and_evaluate


class SegmentationAssistedCNN(nn.Module):
    def __init__(self, in_channels: int = 4, meta_features: int = 5, num_classes: int = 6):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_features, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(160 + 24, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x, extra):
        image_features = self.image_encoder(x)
        meta_features = self.meta_encoder(extra)
        return self.classifier(torch.cat([image_features, meta_features], dim=1))


def main() -> None:
    parser = parser_with_defaults("Train segmentation-assisted DoriaNET model")
    args = parser.parse_args()
    train_and_evaluate("cnn_velocity_segmentation_assisted", SegmentationAssistedCNN(), args, mode="segmented")


if __name__ == "__main__":
    main()
