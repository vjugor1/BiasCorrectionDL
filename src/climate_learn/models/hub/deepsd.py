# Local application
from .utils import register

# Third party
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, scale_factor=2):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=False
        )
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(
            num_features, num_features // 2, kernel_size=5, padding=2
        )
        self.conv3 = nn.Conv2d(
            num_features // 2, out_channels, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
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

        # Calculate the number of SRCNN layers needed
        num_srcnn_layers = self.calculate_num_layers(scale)

        # First SRCNN layer does not take elevation input
        self.srcnn_layers.append(SRCNN(in_channels, out_channels, num_features))
        current_in_channels = out_channels

        # Remaining SRCNN layers take elevation input
        for _ in range(num_srcnn_layers - 1):
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
        if len(self.srcnn_layers) == 1:
            x = self.srcnn_layers[0](x)
        else:
            x = self.srcnn_layers[0](x)
            for srcnn in self.srcnn_layers[1:]:
                x = torch.cat([x, elevation], dim=1)
                x = srcnn(x)
        return x
