import torch
import torch.nn as nn
import math
from .utils import register


@register("ynet")
class YNet30(nn.Module):
    """
    YNet30 architecture adapted from [1].
    YNet30 is a deep learning model tailored for downscaling applications with climatology integration.

    References:
    [1]: https://github.com/yuminliu/Downscaling
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=15,
        num_features=64,
        scale=2,
        use_climatology=True,
    ):
        super(YNet30, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.scale = scale
        self.use_climatology = use_climatology
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=self.scale, mode="bilinear", align_corners=False
        )

        self.conv_layers = self._make_conv_layers()
        self.deconv_layers = self._make_deconv_layers()
        self.subpixel_conv_layer = self._make_subpixel_conv_layer()
        
        self.climatology_channels = self.output_channels + 1

        if self.use_climatology:
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(
                    2 * self.input_channels + self.climatology_channels, # input_channels + input_channels (upsampled) +
                                                    # climatology_channels (same as output_channels) + elevation channel
                    self.num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.num_features,
                    self.output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

    def _make_conv_layers(self):
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    self.input_channels,
                    self.num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(self.num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.num_features, self.num_features, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*layers)

    def _make_deconv_layers(self):
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.num_features, self.num_features, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.num_features, self.num_features, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.num_features,
                    self.num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.num_features,
                    self.input_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )
        return nn.Sequential(*layers)

    def _make_subpixel_conv_layer(self):
        return nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            self.upsample,
            nn.Conv2d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_aux=None):
        residual = x
        residual_up = self.upsample(x)

        # Pass through convolutional layers
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(
                self.num_layers / 2
            ) - 1:
                conv_feats.append(x)

        # Pass through deconvolutional layers
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        # Add the residual connection
        x = x + residual
        x = self.relu(x)

        # Pass through subpixel convolution layer
        x = self.subpixel_conv_layer(x)

        # If use_climatology is True, apply the fusion layer with additional input (x_aux)
        if self.use_climatology and (x_aux is not None):
            # Concatenate with upsampled input
            x = torch.cat([x, residual_up], dim=1)
            # Concatenate with aux data input
            x = torch.cat([x, x_aux], dim=1)
            # Pass through fusion layer
            x = self.fusion_layer(x)  # [Nbatch,Nchannel,Nlat,Nlon]

        return x
