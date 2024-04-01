# Local application
from .utils import register

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F


@register("unet-upsampling")
class UnetUpsampling(nn.Module):
    """Unet upsampling adapted from [1].
    Unet upsampling adapted for specific use-case with size parameter for climate variable maps.

    References:
    [1]: https://github.com/milesial/Pytorch-UNet
    """
    def __init__(self, size, in_channels, bilinear=True):
        super().__init__()
        self.size = size
        if bilinear:
            self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    
    This module performs two sets of operations in sequence:
    1. A convolution that expands or contracts the number of channels,
    2. Batch normalization to stabilize learning and reduce internal covariate shift,
    3. ReLU activation for introducing non-linearity,
    repeated twice.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
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

    def forward(self, x):
        return self.double_conv(x)