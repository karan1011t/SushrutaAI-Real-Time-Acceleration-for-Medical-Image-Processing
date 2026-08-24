"""
Lightweight 2D U-Net Model Architecture for Medical Image Segmentation.

Designed for 2D binary segmentation on MRI slices (e.g. BraTS FLAIR).
Input: (B, 1, 256, 256) -> Output: (B, 1, 256, 256) raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Spatial dimensions alignment
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Lightweight 2D U-Net Model Architecture.

    Parameters:
        in_channels (int): Number of input channels (1 for FLAIR MRI).
        out_channels (int): Number of output channels (1 for binary tumor mask logits).
        base_channels (int): Base feature channel count (default: 32).
        bilinear (bool): Whether to use bilinear upsampling or transposed convolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        c = base_channels

        self.inc = DoubleConv(in_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(c * 8, (c * 16) // factor)
        
        self.up1 = UpBlock(c * 16, (c * 8) // factor, bilinear)
        self.up2 = UpBlock(c * 8, (c * 4) // factor, bilinear)
        self.up3 = UpBlock(c * 4, (c * 2) // factor, bilinear)
        self.up4 = UpBlock(c * 2, c, bilinear)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_summary(self) -> Dict[str, Any]:
        """Returns model summary details."""
        return {
            "model_name": "UNet2D",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "total_parameters": count_parameters(self)
        }


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    dummy_input = torch.randn(2, 1, 256, 256)
    out = model(dummy_input)
    print(f"[TEST] UNet2D instantiated successfully.")
    print(f"       Input shape: {dummy_input.shape}")
    print(f"       Output shape: {out.shape}")
    print(f"       Total trainable parameters: {count_parameters(model):,}")
