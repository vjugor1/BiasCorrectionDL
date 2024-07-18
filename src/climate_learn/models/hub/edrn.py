"""
Enhanced Deep Residual Networks for Single Image Super-Resolution (2017) by Lim et al.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local application
from .utils import register


class ResidualBlock(nn.Module):
    """A basic Residual Block used in EDSR."""

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x) * 0.1



@register("edrn")
class EDRN(nn.Module):
    def __init__(
    self,
    in_channels,
    out_channels,
    channels=256,
    nr_blocks=32,
    scale=2,
): 
        super().__init__()
        
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.channels = channels
        self.nr_blocks = nr_blocks
        self.scale = scale

        self.input_layer = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.input_channels, self.channels, kernel_size=3)
        )
        
        residual_layers = [ResidualBlock(self.channels)] * self.nr_blocks
        self.residual_layers = nn.Sequential(*residual_layers)

        self.upscale = self._make_upscale_module(self.scale)

        self.output_layer = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, self.output_channels, kernel_size=3)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.0001)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _make_upscale_module(self, scale_factor):
        layers = []
        for _ in range(int(scale_factor // 2)):
            layers.append(nn.ReplicationPad2d(1))
            layers.append(nn.Conv2d(self.channels, self.channels * 4, kernel_size=3))
            layers.append(nn.PixelShuffle(2))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_hat = self.input_layer((x - mean) / std)
        x_hat = x_hat + self.residual_layers(x_hat) * 0.1
        x_hat = self.upscale(x_hat)
        return self.output_layer(x_hat) * std + mean