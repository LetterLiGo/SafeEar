# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""MS-STFT discriminator, provided here for reference."""
import typing as tp

import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import einops
from torch.nn import AvgPool1d
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]

CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm',
    'time_group_norm'
])

class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self,
                 normalized_shape: tp.Union[int, tp.List[int], torch.Size],
                 **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


def apply_parametrization_norm(module: nn.Module,
                               norm: str='none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module,
                    causal: bool=False,
                    norm: str='none',
                    **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()
    
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self,
                 *args,
                 causal: bool=False,
                 norm: str='none',
                 norm_kwargs: tp.Dict[str, tp.Any]={},
                 **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self,
                 *args,
                 norm: str='none',
                 norm_kwargs: tp.Dict[str, tp.Any]={},
                 **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(
            self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def get_2d_padding(kernel_size: tp.Tuple[int, int],
                   dilation: tp.Tuple[int, int]=(1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, (
        (kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_fft: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 max_filters: int=1024,
                 filters_scale: int=1,
                 kernel_size: tp.Tuple[int, int]=(3, 9),
                 dilations: tp.List=[1, 2, 4],
                 stride: tp.Tuple[int, int]=(1, 2),
                 normalized: bool=True,
                 norm: str='weight_norm',
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm))
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        # print('x ', x.shape)
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        # print('z ', z.shape)
        z = torch.cat([z.real, z.imag], dim=1)
        # print('cat_z ', z.shape)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            # print('z i', i, z.shape)
            fmap.append(z)
        z = self.conv_post(z)
        # print('logit ', z.shape)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_ffts: tp.List[int]=[1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int]=[256, 512, 128, 64, 32],
                 win_lengths: tp.List[int]=[1024, 2048, 512, 256, 128],
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                win_length=win_lengths[i],
                hop_length=hop_lengths[i],
                **kwargs) for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class DiscriminatorP(torch.nn.Module):
    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False,
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList([
            NormConv2d(
                1,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(
                32,
                32, (kernel_size, 1), (stride, 1),
                padding=(get_padding(5, 1), 0)),
            NormConv2d(32, 32, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = NormConv2d(32, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    

class DiscriminatorS(torch.nn.Module):
    def __init__(self,
                 use_spectral_norm=False,
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super(DiscriminatorS, self).__init__()
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList([
            NormConv1d(1, 32, 15, 1, padding=7),
            NormConv1d(32, 32, 41, 2, groups=4, padding=20),
            NormConv1d(32, 32, 41, 2, groups=16, padding=20),
            NormConv1d(32, 32, 41, 4, groups=16, padding=20),
            NormConv1d(32, 32, 41, 4, groups=16, padding=20),
            NormConv1d(32, 32, 41, 1, groups=16, padding=20),
            NormConv1d(32, 32, 5, 1, padding=2),
        ])
        self.conv_post = NormConv1d(32, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
