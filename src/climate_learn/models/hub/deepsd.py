# Local application
from .utils import register

# Third party
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(
            num_features, num_features // 2, kernel_size=5, padding=2
        )
        self.conv3 = nn.Conv2d(
            num_features // 2, out_channels, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


@register("deepsd")
class DeepSD(nn.Module):
    """
    DeepSD architecture adapted for downscaling applications with auxiliary elevation data.

    References:
    [1]: https://github.com/tjvandal/deepsd
    """

    def __init__(self, in_channels, out_channels, num_features=64, scale=2):
        super(DeepSD, self).__init__()
        self.scale = scale
        self.srcnn_layers = nn.ModuleList()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # Calculate the number of SRCNN layers needed
        num_srcnn_layers = self.calculate_num_layers(scale)
        current_in_channels = in_channels

        # SRCNN layers take elevation input
        for _ in range(num_srcnn_layers):
            self.srcnn_layers.append(
                SRCNN(current_in_channels + 1, out_channels, num_features)
            )
            current_in_channels = out_channels

    def calculate_num_layers(self, scale):
        num_layers = 0
        while scale > 1:
            scale //= 2
            num_layers += 1
        return num_layers

    def forward(self, x, elevation=None):
        if elevation is None or len(elevation) != len(self.srcnn_layers):
            raise ValueError(f"Elevation data must be provided as a list with an entry for each SRCNN layer, expected length {len(self.srcnn_layers)}, got {len(elevation) if elevation is not None else 'None'}.")

        for i, srcnn in enumerate(self.srcnn_layers):
            x = self.upsample(x)
            if elevation[i].shape[2:] != x.shape[2:]:
                raise ValueError(
                    f"Elevation tensor at index {i} with shape {elevation[i].shape} does not match the spatial dimensions of the upscaled input with shape {x.shape}."
                )
            x = torch.cat([x, elevation[i]], dim=1)
            x = srcnn(x)
        return x
