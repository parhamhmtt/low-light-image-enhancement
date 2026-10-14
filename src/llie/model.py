"""Residual U-Net used for paired low-light image enhancement."""
from __future__ import annotations
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.net = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat((x, skip), dim=1))


class LowLightUNet(nn.Module):
    """Encoder-decoder U-Net with residual blocks and sigmoid RGB output.

    Args:
        base_channels: Width of the first encoder stage. Must match the value
            used when the checkpoint was trained (commonly 32 or 64).
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2, 2)
        self.enc3 = ConvBlock(c * 2, c * 4, 2)
        self.enc4 = ConvBlock(c * 4, c * 8, 2)
        self.bot = nn.Sequential(ConvBlock(c * 8, c * 16, 2), ResBlock(c * 16), ResBlock(c * 16))
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.head = nn.Sequential(nn.Conv2d(c, 3, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        x = self.bot(e4)
        x = self.dec4(x, e4); x = self.dec3(x, e3); x = self.dec2(x, e2); x = self.dec1(x, e1)
        return self.head(x)
