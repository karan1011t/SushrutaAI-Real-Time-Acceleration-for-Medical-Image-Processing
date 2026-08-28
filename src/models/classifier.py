"""
SUSHRUTA — Binary Tumor Classifier Model Definition.
Lightweight 2D CNN architecture for binary slice classification (Tumor vs No Tumor).
"""

import torch
import torch.nn as nn


class BinaryTumorClassifier(nn.Module):
    """
    Lightweight 2D Convolutional Neural Network for Binary Tumor Classification.

    Input:  (B, 1, 256, 256) FLAIR MRI Slice
    Output: (B, 1) Raw logit (Class 0: No Tumor, Class 1: Tumor)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.features = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 128x128 -> 64x64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64x64 -> 32x32
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 32x32 -> 16x16
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BinaryTumorClassifier(in_channels=1, base_channels=32)
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"[TEST] Classifier Input shape:  {x.shape}")
    print(f"[TEST] Classifier Output shape: {out.shape}")
    print(f"[TEST] Total parameters:       {count_parameters(model):,}")
