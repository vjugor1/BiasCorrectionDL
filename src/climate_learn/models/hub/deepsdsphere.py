# Local application
from .utils import register

# Third party
import torch
import torch.nn as nn
from .components.sphere_cnn_blocks import SphereConv2d, SphereConvTranspose2d
from torch_harmonics.convolution import DiscreteContinuousConvS2, DiscreteContinuousConvTransposeS2


class SRCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, in_shape, out_shape):
        super(SRCNN, self).__init__()
        # self.conv1 = SphereConv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.conv1 = DiscreteContinuousConvS2(in_channels=in_channels, out_channels=num_features, in_shape=in_shape, out_shape=out_shape, kernel_shape=9)
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


@register("deepsdsphere")
class DeepSDSphere(nn.Module):
    """
    DeepSD architecture adapted for downscaling applications with auxiliary elevation data.

    References:
    [1]: https://github.com/tjvandal/deepsd
    """

    def __init__(self, in_channels, out_channels, in_shape, out_shape, num_features=64, scale=2):
        super(DeepSDSphere, self).__init__()
        self.scale = scale
        self.srcnn_layers = nn.ModuleList()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # Calculate the number of SRCNN layers needed
        num_srcnn_layers = self.calculate_num_layers(scale)
        current_in_channels = in_channels

        # SRCNN layers take elevation input
        in_shape_ = in_shape
        out_shape_ = (in_shape_[0], in_shape_[1])
        for _ in range(num_srcnn_layers):
            in_shape_ = (out_shape_[0] * 2, out_shape_[1] * 2)
            out_shape_ = in_shape_
            self.srcnn_layers.append(
                SRCNN(current_in_channels + 1, out_channels, num_features, in_shape_, out_shape_)
            )
            current_in_channels = out_channels
        assert (out_shape_[0] == out_shape[0]) and (out_shape_[1] == out_shape[1])
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
