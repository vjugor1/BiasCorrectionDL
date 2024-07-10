# Standard Library
from typing import Callable, Optional, Union

# Local application
from .utils import MetricsMetaInfo, register
from .functional import *

# Third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


class Metric:
    """Parent class for all ClimateLearn metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        r"""
        .. highlight:: python

        :param aggregate_only: If false, returns both the aggregate and
            per-channel metrics. Otherwise, returns only the aggregate metric.
            Default is `False`.
        :type aggregate_only: bool
        :param metainfo: Optional meta-information used by some metrics.
        :type metainfo: MetricsMetaInfo|None
        """
        self.aggregate_only = aggregate_only
        self.metainfo = metainfo

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred: The predicted value(s).
        :type pred: torch.Tensor
        :param target: The ground truth target value(s).
        :type target: torch.Tensor

        :return: A tensor. See child classes for specifics.
        :rtype: torch.Tensor
        """
        raise NotImplementedError()


class LatitudeWeightedMetric(Metric):
    """Parent class for latitude-weighted metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        lat_weights = np.cos(np.deg2rad(self.metainfo.lat))
        lat_weights = lat_weights / lat_weights.mean()
        lat_weights = torch.from_numpy(lat_weights).view(1, 1, -1, 1)
        self.lat_weights = lat_weights

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts latitude weights to the same device as `pred`.
        """
        self.lat_weights = self.lat_weights.to(device=pred.device)


class ClimatologyBasedMetric(Metric):
    """Parent class for metrics that use climatology."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        climatology = self.metainfo.climatology
        climatology = climatology.unsqueeze(0)
        self.climatology = climatology

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts climatology to the same device as `pred`.
        """
        self.climatology = self.climatology.to(device=pred.device)


class TransformedMetric:
    """Class which composes a transform and a metric."""

    def __init__(self, transform: Callable, metric: Metric):
        self.transform = transform
        self.metric = metric
        self.name = metric.name

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> None:
        pred = self.transform(pred)
        target = self.transform(target)
        return self.metric(pred, target)


@register("mse")
class MSE(Metric):
    """Computes standard mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mse(pred, target, self.aggregate_only)


@register("lat_mse")
class LatWeightedMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        return mse(pred, target, self.aggregate_only, self.lat_weights)


@register("rmse")
class RMSE(Metric):
    """Computes standard root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, mask)
        return rmse(pred, target, self.aggregate_only)


@register("lat_rmse")
class LatWeightedRMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, self.lat_weights, mask)
        return rmse(pred, target, self.aggregate_only, self.lat_weights)


@register("acc")
class ACC(ClimatologyBasedMetric):
    """
    Computes standard anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return acc(pred, target, self.climatology, self.aggregate_only, mask)
        return acc(pred, target, self.climatology, self.aggregate_only)


@register("lat_acc")
class LatWeightedACC(LatitudeWeightedMetric, ClimatologyBasedMetric):
    """
    Computes latitude-weighted anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        LatitudeWeightedMetric.__init__(self, *args, **kwargs)
        ClimatologyBasedMetric.__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        LatitudeWeightedMetric.cast_to_device(self, pred)
        ClimatologyBasedMetric.cast_to_device(self, pred)
        if mask is not None:
            return acc(
                pred,
                target,
                self.climatology,
                self.aggregate_only,
                self.lat_weights,
                mask,
            )
        return acc(
            pred, target, self.climatology, self.aggregate_only, self.lat_weights
        )


@register("pearson")
class Pearson(Metric):
    """
    Computes the Pearson correlation coefficient, based on
    https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/10
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            Pearson correlation coefficient, and the preceding elements are the
            channel-wise Pearson correlation coefficients.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return pearson(pred, target, self.aggregate_only)


@register("mean_bias")
class MeanBias(Metric):
    """Computes the standard mean bias."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate mean
            bias, and the preceding elements are the channel-wise mean bias.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mean_bias(pred, target, self.aggregate_only)
@register("perceptual")
class VGGLoss(Metric):
    """Computes perceptual loss with VGG16"""
    def __init__(self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None, device: torch.cuda.device = 1):
        super().__init__(aggregate_only, metainfo)
        vgg_features = torchvision.models.vgg16(pretrained=True).features
        modules = [m for m in vgg_features]
        
        # if conv_index == '22':
        #     self.vgg = nn.Sequential(*modules[:8])
        # elif conv_index == '54':
        #     self.vgg = nn.Sequential(*modules[:35])
        self.vgg = nn.Sequential(*modules[:4]).to(device)
        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
    def vgg_over_triplicated_channels(self, x, reducer = sum):
        n_channels = x.shape[1]
        return reducer([self.vgg(x[:, c, :, :].unsqueeze(1).repeat(1, 3, 1, 1).float()) for c in range(n_channels)])
    def gram(self, x):
        b, c, h, w = x.size()
        g = torch.bmm(x.view(b, c, h*w) / math.sqrt(h*w), x.view(b, c, h*w).transpose(1,2) / math.sqrt(h*w))
        return g#.div(h*w)
    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        vgg_sr = self.vgg_over_triplicated_channels(pred.float())

        with torch.no_grad():
            vgg_hr = self.vgg_over_triplicated_channels(target.float().detach())#self.vgg(target.float().detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)
        gram_loss = F.mse_loss(self.gram(vgg_sr), self.gram(vgg_hr))
        # gram_loss = 0
        # for f, t in zip(vgg_sr, vgg_hr):
        #     gram_loss += F.mse_loss(self.gram(f), self.gram(t))
        
        return mse(pred, target, self.aggregate_only) + 0.01 * loss # + 0.1 * gram_loss

@register("edge")
class edge_loss(Metric):
    def __init__(self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None, device: torch.cuda.device = 1, dtype: torch.dtype = torch.half, n_chan: int = 4):
        super().__init__(aggregate_only, metainfo)
        x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
        self.weights_x = torch.tensor(x_filter, requires_grad=False, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(1, n_chan, 1, 1)
        self.weights_y = torch.tensor(y_filter, requires_grad=False, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(1, n_chan, 1, 1)
    def __call__(self, out, target):
        

        # convx.weight = nn.Parameter(weights_x)
        # convy.weight = nn.Parameter(weights_y)

        g1_x = F.conv2d(out, self.weights_x, padding=1, stride=1)
        g2_x = F.conv2d(target, self.weights_x, padding=1, stride=1)
        g1_y = F.conv2d(out, self.weights_y, padding=1, stride=1)
        g2_y = F.conv2d(target, self.weights_y, padding=1, stride=1)

        g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
        g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

        return mse(out, target, self.aggregate_only) + 0.01 * torch.mean((g_1 - g_2).pow(2))

@register("mse_img_freq")
class OriFreqMSE(Metric):
    """Computes MSE loss within original and frequency domain. For frequency it is both amplitude and phase"""
    # def __init__(self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None):
    #     super().__init__(aggregate_only, metainfo)
    def get_amp_phase(self, img):
        fft_im = torch.fft.rfft2(img, norm='ortho')
        # fft_im: size should be bxcxhxwx2
        fft_amp = torch.abs(fft_im) # this is the amplitude
        # eps = 1e-7
        # nudge = (torch.real(fft_im[:,:,:,:]) <= eps) * eps
        # fft_pha = torch.atan2( torch.imag(fft_im[:,:,:,:]), torch.real(fft_im[:,:,:,:]) + nudge) # this is the phase
        return fft_amp#, fft_pha
    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        # amp_pred, phase_pred = self.get_amp_phase(pred)
        amp_pred = self.get_amp_phase(pred)
        # amp_target, phase_target = self.get_amp_phase(target)
        amp_target = self.get_amp_phase(target)
        if amp_pred.isnan().any() or amp_target.isnan().any():
            raise ValueError
        return F.mse_loss(amp_pred, amp_target) + F.mse_loss(pred, target)# + F.mse_loss(phase_pred, phase_target)

# @register("perceptual")
# class VGGLoss(Metric):
#     """Computes perceptual loss with VGG16"""
#     def __init__(self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None, device: torch.cuda.device = 1):
#         super().__init__(aggregate_only, metainfo)
#         self.vgg = VGG19Features().to(device).eval()
#         self.criterion = nn.MSELoss()
#         # vgg_features = torchvision.models.vgg19(pretrained=True).features
#         # modules = [m for m in vgg_features]
        
#         # # if conv_index == '22':
#         # #     self.vgg = nn.Sequential(*modules[:8])
#         # # elif conv_index == '54':
#         # #     self.vgg = nn.Sequential(*modules[:35])
#         # # self.vgg = nn.Sequential(*modules[:35]).to(device)
#         # self.vgg = nn.Sequential(*(modules[:8] + modules[-8:])).to(device)
#         # # vgg_mean = (0.485, 0.456, 0.406)
#         # # vgg_std = (0.229, 0.224, 0.225)
#         # #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
#         # self.vgg.requires_grad = False
#     def vgg_over_triplicated_channels(self, x, reducer = None):
#         n_channels = x.shape[1]
#         if reducer is not None:
#             return reducer([self.vgg(x[:, c, :, :].unsqueeze(1).repeat(1, 3, 1, 1).float()) for c in range(n_channels)])
#         else:
#             return [self.vgg(x[:, c, :, :].unsqueeze(1).repeat(1, 3, 1, 1).float()) for c in range(n_channels)]
#     def __call__(
#         self,
#         pred: Union[torch.FloatTensor, torch.DoubleTensor],
#         target: Union[torch.FloatTensor, torch.DoubleTensor],
#     ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
#         x_vgg = self.vgg_over_triplicated_channels(pred.float())

#         # with torch.no_grad():
#         y_vgg = self.vgg_over_triplicated_channels(target.float().detach())#self.vgg(target.float().detach())

#         loss = 0
#         for channel in range(len(x_vgg)):
#             for i in range(len(x_vgg[channel])):
#                 loss += self.criterion(x_vgg[channel][i], y_vgg[channel][i])
        
#         return 0.1 * loss + mse(pred, target, self.aggregate_only)

# class VGG19Features(nn.Module):
#     def __init__(self):
#         super(VGG19Features, self).__init__()
#         vgg19 = torchvision.models.vgg19(pretrained=False).features
#         self.slice1 = nn.Sequential()
#         self.slice2 = nn.Sequential()
#         self.slice3 = nn.Sequential()
#         self.slice4 = nn.Sequential()
#         self.slice5 = nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg19[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg19[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg19[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg19[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg19[x])
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         h = self.slice1(x)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h
#         h = self.slice5(h)
#         h_relu5_4 = h
#         return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_4

@register("mae")
class MAE(Metric):
    """Computes MAE loss."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise L1 losses.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mae(pred, target, self.aggregate_only)
    

@register("bce")
class BCE(Metric):
    """Computes binary cross entropy loss."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            BCE, and the preceding elements are the channel-wise BCEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return bce(pred, target, self.aggregate_only)