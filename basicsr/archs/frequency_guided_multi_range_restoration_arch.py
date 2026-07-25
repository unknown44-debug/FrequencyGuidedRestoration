# -*- coding: utf-8 -*-
"""Frequency-guided multi-range video restoration architecture.

The implementation is organized according to the paper methodology:

1. Global-High-frequency Prompt Interaction (GHPI)
2. Frequency-guided Second-order Propagation (FGSP)
3. Bidirectional Multi-Range Temporal Routing (BMTR)
4. Progressive gated reconstruction

The file contains the complete final architecture and no longer depends on a
separate prompt/alignment architecture file.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from einops import rearrange
except ImportError as exc:  # pragma: no cover
    raise ImportError('This architecture requires einops. Please install einops.') from exc

from basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.spynet_arch import SpyNet as MotionEstimator
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.ops import deform_conv2d


class ModulatedDeformConv2d(nn.Module):
    """MMCV-compatible DCNv2 parameter container backed by torchvision.

    Keeping this small compatibility layer avoids requiring a separately
    compiled MMCV package while preserving checkpoint parameter names and
    deformable-convolution behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups.')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        bound = 1.0 / math.sqrt(
            in_channels * kernel_size[0] * kernel_size[1]
        )
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def modulated_deform_conv2d(
    input_tensor: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    deform_groups: int,
) -> torch.Tensor:
    """Call torchvision DCNv2; groups are inferred from tensor shapes."""
    del groups, deform_groups
    return deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )


class ResidualBlocksWithInputConv(nn.Module):
    """Input projection followed by BasicSR residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        num_blocks: int = 30,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(
                ResidualBlockNoBN,
                num_blocks,
                num_feat=out_channels,
            ),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.main(feature)


def reconstruction_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
) -> nn.Conv2d:
    """Convolution used by the progressive reconstruction head."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


class _ReconstructionChannelAttention(nn.Module):
    """Channel attention retained for legacy reconstruction checkpoints."""

    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        # The experimental reconstruction used no bottleneck in this block.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class ReconstructionChannelBlock(nn.Module):
    """Two-convolution residual channel-attention block."""

    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool,
        act: nn.Module,
    ) -> None:
        super().__init__()
        del reduction
        self.body = nn.Sequential(
            reconstruction_conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            reconstruction_conv(n_feat, n_feat, kernel_size, bias=bias),
        )
        self.CA = _ReconstructionChannelAttention(n_feat, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.CA(self.body(x))


class ReconstructionUpsampler(nn.Module):
    """Convolution followed by pixel shuffle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        upsample_kernel: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class SkipGuidedUpsampler(nn.Module):
    """Bilinear 2x upsampling followed by a projected skip addition."""

    def __init__(self, in_channels: int, s_factor: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                in_channels + s_factor,
                in_channels,
                1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        return x + skip


class _ReconstructionLayerNorm2d(nn.Module):
    """Per-pixel channel normalization with legacy parameter names."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            x * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class _DepthwiseRepConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, bias: bool) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=bias,
        )
        self.conv_2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_1(x) + self.conv_2(x) + x


class _DepthwiseResidualConv(nn.Module):
    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.conv_2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(x) + x


class _SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _SigmoidGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class _ShiftChannelBlock(nn.Module):
    """Channel block used after spatial-temporal feature shifting."""

    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        bias: bool,
        add_channel: int,
    ) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.add_channel = add_channel
        self.conv1 = nn.Conv2d(
            add_channel,
            add_channel,
            3,
            padding=1,
            groups=add_channel,
            bias=bias,
        )
        self.norm = _ReconstructionLayerNorm2d(n_feat + add_channel)
        self.body = nn.Sequential(
            reconstruction_conv(
                n_feat + add_channel, 2 * n_feat, 1, bias=bias
            ),
            _DepthwiseResidualConv(2 * n_feat, bias),
            _SimpleGate(),
            _DepthwiseRepConv(n_feat, kernel_size, bias),
            reconstruction_conv(n_feat, 2 * n_feat, 1, bias=bias),
            _SigmoidGate(),
            _ReconstructionChannelAttention(n_feat, bias=bias),
            reconstruction_conv(n_feat, n_feat, 1, bias=bias),
        )
        self.beta = nn.Parameter(torch.zeros(1, n_feat, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut, shifted = x[:, :self.n_feat], x[:, self.n_feat:]
        shifted = self.conv1(shifted)
        residual = self.body(
            self.norm(torch.cat([shortcut, shifted], dim=1))
        )
        return shortcut + residual * self.beta


class _PostShiftChannelBlock(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, bias: bool) -> None:
        super().__init__()
        self.norm = _ReconstructionLayerNorm2d(n_feat)
        self.body = nn.Sequential(
            reconstruction_conv(n_feat, 2 * n_feat, 1, bias=bias),
            _DepthwiseResidualConv(2 * n_feat, bias),
            _SimpleGate(),
            _DepthwiseRepConv(n_feat, kernel_size, bias),
            reconstruction_conv(n_feat, 2 * n_feat, 1, bias=bias),
            _SigmoidGate(),
            _ReconstructionChannelAttention(n_feat, bias=bias),
            reconstruction_conv(n_feat, n_feat, 1, bias=bias),
        )
        self.beta = nn.Parameter(torch.zeros(1, n_feat, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(self.norm(x)) * self.beta


class TemporalShiftReconstructionBlock(nn.Module):
    """Four-stage temporal/channel shift reconstruction block.

    The first tensor dimension represents time when this block is called by the
    reconstruction head. The implementation keeps the parameter layout of the
    earlier experimental block so existing checkpoints remain loadable.
    """

    _SHIFT_OFFSETS = (
        (8, 8), (8, 4), (8, 0), (8, -4), (8, -8),
        (-8, 8), (-8, 4), (-8, 0), (-8, -4), (-8, -8),
        (4, 8), (4, -8), (0, 8), (0, -8), (-4, 8), (-4, -8),
        (4, 4), (4, 0), (4, -4), (0, 4), (0, -4),
        (-4, 4), (-4, 0), (-4, -4),
    )

    def __init__(
        self,
        n_features: int,
        kernel_size: int,
        reduction: int,
        bias: bool = False,
        scale_unetfeats: int = 48,
    ) -> None:
        super().__init__()
        del reduction, scale_unetfeats
        self.number = n_features // 16
        if self.number < 1:
            raise ValueError('Temporal shift reconstruction needs at least 16 channels.')
        shifted_channels = 8 * self.number

        def make_stage() -> nn.Sequential:
            return nn.Sequential(
                _ShiftChannelBlock(
                    n_features,
                    kernel_size=5,
                    bias=bias,
                    add_channel=shifted_channels,
                ),
                _PostShiftChannelBlock(
                    n_features,
                    kernel_size=5,
                    bias=bias,
                ),
            )

        self.encoder_level1 = make_stage()
        self.encoder_level1_1 = make_stage()
        self.encoder_level1_2 = make_stage()
        self.encoder_level1_3 = make_stage()

    @staticmethod
    def _translate_without_wrap(
        x: torch.Tensor,
        dy: int,
        dx: int,
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        h, w = x.shape[-2:]
        if abs(dy) >= h or abs(dx) >= w:
            return out
        src_y0 = max(-dy, 0)
        src_y1 = h - max(dy, 0)
        src_x0 = max(-dx, 0)
        src_x1 = w - max(dx, 0)
        dst_y0 = max(dy, 0)
        dst_y1 = h - max(-dy, 0)
        dst_x0 = max(dx, 0)
        dst_x1 = w - max(-dx, 0)
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[
            :, :, src_y0:src_y1, src_x0:src_x1
        ]
        return out

    def _spatial_shift(self, x: torch.Tensor) -> torch.Tensor:
        n2 = (self.number - 1) // 2
        n1 = self.number - 2 * n2
        widths = [n2] * 16 + [n1] * 8
        parts: List[torch.Tensor] = []
        start = 0
        for width, (dy, dx) in zip(widths, self._SHIFT_OFFSETS):
            end = start + width
            if width:
                parts.append(
                    self._translate_without_wrap(x[:, start:end], dy, dx)
                )
            start = end
        if start != x.size(1):
            raise RuntimeError(
                f'Shift channel partition covers {start} of {x.size(1)} channels.'
            )
        return torch.cat(parts, dim=1)

    def _channel_shift(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        shift = c // 2
        if reverse:
            shift = -shift
        mixed = torch.roll(x.reshape(1, b * c, h, w), shift, 1)
        mixed = mixed.reshape(b, c, h, w)
        shifted = (
            mixed[:, -8 * self.number:]
            if reverse
            else mixed[:, :8 * self.number]
        )
        return torch.cat([mixed, self._spatial_shift(shifted)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        reverse: int = 0,
    ) -> torch.Tensor:
        del reverse
        x = self.encoder_level1(self._channel_shift(x))
        x = self.encoder_level1_1(self._channel_shift(x, reverse=True))
        x = self.encoder_level1_2(self._channel_shift(x))
        return self.encoder_level1_3(self._channel_shift(x, reverse=True))


def initialize_constant(module: nn.Module, val: float, bias: float = 0.0) -> None:
    """Initialize a module with constants without depending on mmcv.cnn."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def warp_feature(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp NCHW tensor ``x`` with N2HW flow using BasicSR flow_warp."""
    return flow_warp(x, flow.permute(0, 2, 3, 1))


def compose_motion(flow_ab: torch.Tensor, flow_bc: torch.Tensor) -> torch.Tensor:
    """Compose a->b and b->c flows into a->c on coordinate system a."""
    return flow_ab + warp_feature(flow_bc, flow_ab)


def resize_motion(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize an N2HW flow field and scale displacement values accordingly."""
    h_old, w_old = flow.shape[-2:]
    h_new, w_new = size
    if (h_old, w_old) == (h_new, w_new):
        return flow
    flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
    flow = flow.clone()
    flow[:, 0, :, :] *= float(w_new) / float(w_old)
    flow[:, 1, :, :] *= float(h_new) / float(h_old)
    return flow


def resize_motion_sequence(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a B,T,2,H,W flow sequence to feature scale."""
    if flow.numel() == 0:
        return flow
    b, t, c, h, w = flow.shape
    flow_flat = flow.reshape(b * t, c, h, w)
    flow_flat = resize_motion(flow_flat, size)
    return flow_flat.view(b, t, c, size[0], size[1])


def bidirectional_motion_consistency(
    flow_fw: torch.Tensor,
    flow_bw: torch.Tensor,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute coordinate-correct forward-backward consistency.

    Args:
        flow_fw: target -> source flow, shape [B,2,H,W].
        flow_bw: source -> target flow, shape [B,2,H,W].

    Returns:
        residual_normalized: normalized FB residual, shape [B,2,H,W].
        soft_confidence: exp(-error / threshold), shape [B,1,H,W].
        hard_valid: binary valid map, shape [B,1,H,W].
    """
    flow_bw_warped = warp_feature(flow_bw, flow_fw)
    residual = flow_fw + flow_bw_warped

    error_sq = residual.square().sum(dim=1, keepdim=True)
    motion_sq = (
        flow_fw.square().sum(dim=1, keepdim=True)
        + flow_bw_warped.square().sum(dim=1, keepdim=True)
    )
    threshold = alpha1 * motion_sq + alpha2

    hard_valid = (error_sq < threshold).to(flow_fw.dtype)
    soft_confidence = torch.exp(-error_sq / (threshold + eps))
    residual_normalized = residual / torch.sqrt(motion_sq + alpha2 + eps)
    return residual_normalized, soft_confidence, hard_valid


class AdaptiveFrequencyMaskGenerator(nn.Module):
    """Sample-adaptive rectangular frequency mask generator.

    ``ste_hard`` gives a hard rectangular forward mask and uses a soft surrogate
    in backward. This keeps the frequency mask interpretable while avoiding the
    empty-mask and non-differentiability issues of direct integer slicing on
    1/4-resolution video features.
    """

    _VALID_MODES = {'ste_hard', 'soft', 'fixed'}

    def __init__(
        self,
        channels: int,
        mask_mode: str = 'ste_hard',
        min_half_ratio: float = 0.05,
        max_half_ratio: float = 0.45,
        softness: float = 0.03,
        fixed_half_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if mask_mode not in self._VALID_MODES:
            raise ValueError(
                f'Unsupported mask_mode={mask_mode}; '
                f'expected one of {sorted(self._VALID_MODES)}.'
            )
        if not (0.0 < min_half_ratio < max_half_ratio <= 0.5):
            raise ValueError(
                'Require 0 < min_half_ratio < max_half_ratio <= 0.5.'
            )
        if softness <= 0.0:
            raise ValueError('softness must be positive.')
        if not (0.0 < fixed_half_ratio <= 0.5):
            raise ValueError('fixed_half_ratio must be in (0, 0.5].')

        hidden = max(channels // 8, 4)
        self.mask_mode = mask_mode
        self.min_half_ratio = float(min_half_ratio)
        self.max_half_ratio = float(max_half_ratio)
        self.softness = float(softness)
        self.fixed_half_ratio = float(fixed_half_ratio)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=False),
        )

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = (
            torch.arange(height, device=device, dtype=dtype)
            - (height - 1) / 2.0
        ).abs() / max(float(height), 1.0)
        x = (
            torch.arange(width, device=device, dtype=dtype)
            - (width - 1) / 2.0
        ).abs() / max(float(width), 1.0)
        return y.view(1, 1, height, 1), x.view(1, 1, 1, width)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f'Expected [B,C,H,W], got {tuple(x.shape)}.')

        b, _, h, w = x.shape
        dtype = x.dtype
        device = x.device

        if self.mask_mode == 'fixed':
            half_ratio = x.new_full((b, 2, 1, 1), self.fixed_half_ratio)
        else:
            raw_ratio = torch.sigmoid(
                self.rate_conv(F.adaptive_avg_pool2d(x, 1))
            )
            half_ratio = self.min_half_ratio + (
                self.max_half_ratio - self.min_half_ratio
            ) * raw_ratio

        half_h = half_ratio[:, 0:1]
        half_w = half_ratio[:, 1:2]
        yy, xx = self._coordinate_grid(h, w, device, dtype)

        soft_y = torch.sigmoid((half_h - yy) / self.softness)
        soft_x = torch.sigmoid((half_w - xx) / self.softness)
        soft_low_mask = soft_y * soft_x

        if self.mask_mode == 'soft':
            low_mask = soft_low_mask
        else:
            hard_low_mask = (soft_low_mask >= 0.5).to(dtype)
            if self.mask_mode == 'ste_hard':
                low_mask = (
                    hard_low_mask.detach()
                    - soft_low_mask.detach()
                    + soft_low_mask
                )
            else:
                low_mask = hard_low_mask

        return low_mask, half_ratio


class HighFrequencyResponseExtractor(nn.Module):
    """Adaptive masking with FFT/IFFT high-frequency spatial response extractor."""

    def __init__(
        self,
        channels: int,
        mask_mode: str = 'ste_hard',
        min_half_ratio: float = 0.05,
        max_half_ratio: float = 0.45,
        softness: float = 0.03,
        fixed_half_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.pre_conv = nn.Conv2d(
            channels, channels, 3, stride=1, padding=1, bias=False
        )
        self.frequency_mask_generator = AdaptiveFrequencyMaskGenerator(
            channels=channels,
            mask_mode=mask_mode,
            min_half_ratio=min_half_ratio,
            max_half_ratio=max_half_ratio,
            softness=softness,
            fixed_half_ratio=fixed_half_ratio,
        )
        self.high_dconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.high_act = nn.GELU()

    def forward(
        self, current_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if current_feat.ndim != 4:
            raise ValueError(
                'current_feat must have shape [B,C,H,W], got '
                f'{tuple(current_feat.shape)}.'
            )
        if current_feat.size(1) != self.channels:
            raise ValueError(
                f'Expected {self.channels} channels, got {current_feat.size(1)}.'
            )

        projected = self.pre_conv(current_feat)

        # Complex FFT/IFFT is kept in float32 for numerical stability.
        fft_input = projected.float()
        spectrum = torch.fft.fft2(fft_input, norm='forward', dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

        low_mask, half_ratio = self.frequency_mask_generator(projected.float())
        high_mask = 1.0 - low_mask
        high_spectrum = spectrum * high_mask

        high = torch.fft.ifft2(
            torch.fft.ifftshift(high_spectrum, dim=(-2, -1)),
            norm='forward',
            dim=(-2, -1),
        ).abs()
        high = high.to(dtype=current_feat.dtype)
        high = self.high_act(self.high_dconv(high))

        aux = {
            'frequency_mask_low_mask_mean': low_mask.detach().mean(),
            'frequency_mask_half_h_ratio_mean': half_ratio[:, 0].detach().mean(),
            'frequency_mask_half_w_ratio_mean': half_ratio[:, 1].detach().mean(),
            'high_frequency_feature_abs_mean': high.detach().abs().mean(),
        }
        return high, aux


class LocalHighFrequencyPromptInteraction(nn.Module):
    """Local high-frequency prompt interaction with padding support.

    Query comes from the global prompt. Key/Value come from the spatial
    high-frequency response. Unlike the old high-frequency prompt path path, this module does not
    perform GAP on the high-frequency feature map.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        win_size: int = 8,
        num_heads: int = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        bias: bool = False,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(
                f'query_dim={query_dim} must be divisible by num_heads={num_heads}.'
            )
        if win_size <= 0:
            raise ValueError('win_size must be positive.')

        self.query_dim = int(query_dim)
        self.kv_dim = int(kv_dim)
        self.win_size = int(win_size)
        self.num_heads = int(num_heads)
        head_dim = query_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * query_dim, bias=qkv_bias)
        self.kv_dwconv = nn.Conv2d(
            kv_dim,
            kv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=kv_dim,
            bias=bias,
        )
        self.project_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        if zero_init:
            nn.init.zeros_(self.project_out.weight)
            if self.project_out.bias is not None:
                nn.init.zeros_(self.project_out.bias)

    def _partition_windows(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B*nW, ws*ws, C]
        b, c, h, w = x.shape
        ws = self.win_size
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws * ws, c)

    def _reverse_windows(
        self,
        windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        channels: int,
    ) -> torch.Tensor:
        # [B*nW, ws*ws, C] -> [B,C,H,W]
        ws = self.win_size
        x = windows.view(
            batch,
            height // ws,
            width // ws,
            ws,
            ws,
            channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch, channels, height, width)

    def forward(
        self,
        query_feature: torch.Tensor,
        key_value_feature: torch.Tensor,
    ) -> torch.Tensor:
        if query_feature.ndim != 4 or key_value_feature.ndim != 4:
            raise ValueError('query_feature and key_value_feature must be [B,C,H,W].')

        b, c, h, w = query_feature.shape
        if c != self.query_dim:
            raise ValueError(
                f'Expected query_feature with {self.query_dim} channels, got {c}.'
            )
        if key_value_feature.size(0) != b:
            raise ValueError('Batch size mismatch between query and key/value.')
        if key_value_feature.size(1) != self.kv_dim:
            raise ValueError(
                f'Expected key_value_feature with {self.kv_dim} channels, '
                f'got {key_value_feature.size(1)}.'
            )

        if key_value_feature.shape[-2:] != (h, w):
            key_value_feature = F.interpolate(
                key_value_feature,
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )

        key_value_feature = self.kv_dwconv(key_value_feature)

        ws = self.win_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            query_feature = F.pad(query_feature, (0, pad_w, 0, pad_h))
            key_value_feature = F.pad(key_value_feature, (0, pad_w, 0, pad_h))

        hp, wp = query_feature.shape[-2:]
        q_windows = self._partition_windows(query_feature)
        kv_windows = self._partition_windows(key_value_feature)

        num_windows_b, num_tokens, _ = q_windows.shape
        head_dim = self.query_dim // self.num_heads

        q = self.q_proj(q_windows)
        q = q.view(num_windows_b, num_tokens, self.num_heads, head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        q = q * self.scale

        kv = self.kv_proj(kv_windows)
        kv = kv.view(num_windows_b, num_tokens, 2, self.num_heads, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = torch.matmul(q, k.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(num_windows_b, num_tokens, self.query_dim)
        out = self.project_out(out)
        out = self._reverse_windows(out, b, hp, wp, self.query_dim)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return out


class GlobalHighFrequencyPromptInteraction(nn.Module):
    """Alignment-only global prompt + local high-frequency prompt injection.

    Global branch:
        x -> GAP -> global prompt bank -> P_G

    Local high-frequency branch:
        current_feat -> FFT high-frequency response H_i
        WPF(Q=P_G, K=H_i, V=H_i) -> Delta P_H

    Final prompt:
        P_F = P_G + high_frequency_scale * Delta P_H

    The high-frequency response is not globally averaged, so the branch is
    preserves spatial high-frequency cues while retaining global degradation conditioning.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        prompt_dim: int = 96,
        prompt_len: int = 5,
        prompt_size: int = 96,
        num_blocks: int = 1,
        ghpi_heads: int = 4,
        interaction_scale_init: float = 0.01,
        use_high_frequency_prompt: bool = True,
        high_frequency_window_size: int = 8,
        high_frequency_zero_init: bool = True,
        frequency_mask_mode: str = 'ste_hard',
        frequency_min_half_ratio: float = 0.05,
        frequency_max_half_ratio: float = 0.45,
        frequency_mask_softness: float = 0.03,
        frequency_fixed_half_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.prompt_dim = int(prompt_dim)
        self.prompt_len = int(prompt_len)
        self.prompt_size = int(prompt_size)
        self.use_high_frequency_prompt = bool(use_high_frequency_prompt)
        self.high_frequency_window_size = int(high_frequency_window_size)

        # Global degradation-conditioned prompt branch.
        self.prompt_param = nn.Parameter(
            torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_proj = nn.Linear(embed_dim, prompt_len)

        # Spatial HF response extractor. Its output is used as local K/V and is
        # not collapsed by GAP.
        if self.use_high_frequency_prompt:
            self.high_frequency_extractor = HighFrequencyResponseExtractor(
                channels=embed_dim,
                mask_mode=frequency_mask_mode,
                min_half_ratio=frequency_min_half_ratio,
                max_half_ratio=frequency_max_half_ratio,
                softness=frequency_mask_softness,
                fixed_half_ratio=frequency_fixed_half_ratio,
            )
            self.local_prompt_interaction = LocalHighFrequencyPromptInteraction(
                query_dim=prompt_dim,
                kv_dim=embed_dim,
                win_size=high_frequency_window_size,
                num_heads=ghpi_heads,
                qkv_bias=True,
                qk_scale=None,
                bias=False,
                zero_init=high_frequency_zero_init,
            )
        self.high_frequency_scale = nn.Parameter(torch.tensor(float(interaction_scale_init)))

        # Post-interaction feature refinement.
        self.conv = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.residual = ResidualBlocksWithInputConv(
            embed_dim + prompt_dim, embed_dim, num_blocks
        )
        self.latest_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        safe = weights.clamp_min(eps)
        return -(safe * safe.log()).sum(dim=1).mean()

    @staticmethod
    def _resize_prompt_to_feature(
        prompt: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        ph, pw = prompt.shape[-2:]
        th, tw = target_hw
        if (ph, pw) == (th, tw):
            return prompt
        # When shrinking prompt bank maps, average pooling is more stable than
        # bilinear interpolation because it preserves prompt-level statistics.
        if th <= ph and tw <= pw:
            return F.adaptive_avg_pool2d(prompt, output_size=(th, tw))
        return F.interpolate(prompt, size=(th, tw), mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor, current_feat: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or current_feat.ndim != 4:
            raise ValueError('x and current_feat must be [B,C,H,W].')
        if x.size(1) != self.embed_dim:
            raise ValueError(f'Expected x with {self.embed_dim} channels, got {x.size(1)}.')
        if current_feat.size(1) != self.embed_dim:
            raise ValueError(
                f'Expected current_feat with {self.embed_dim} channels, '
                f'got {current_feat.size(1)}.'
            )

        b, _, h, w = x.shape

        # Global prompt routing.
        global_embedding = x.mean(dim=(-2, -1))
        global_weights = F.softmax(self.linear_proj(global_embedding), dim=1)
        global_prompt = torch.einsum(
            'bk,kchw->bchw', global_weights, self.prompt_param
        )

        extractor_aux: Dict[str, torch.Tensor] = {}
        if self.use_high_frequency_prompt:
            high_feature, extractor_aux = self.high_frequency_extractor(current_feat)

            # Local high-frequency prompt interaction: Q from global prompt, K/V
            # from spatial HF response. No high_feature.mean is used here.
            global_prompt_hf = self._resize_prompt_to_feature(
                global_prompt, high_feature.shape[-2:]
            )
            delta_prompt = self.local_prompt_interaction(
                query_feature=global_prompt_hf,
                key_value_feature=high_feature,
            )
            if delta_prompt.shape[-2:] != global_prompt.shape[-2:]:
                delta_prompt = F.interpolate(
                    delta_prompt,
                    size=global_prompt.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            fused_prompt = global_prompt + self.high_frequency_scale * delta_prompt
        else:
            delta_prompt = torch.zeros_like(global_prompt)
            fused_prompt = global_prompt

        fused_prompt = F.interpolate(
            fused_prompt, size=(h, w), mode='bilinear', align_corners=False
        )
        fused_prompt = self.conv(fused_prompt)
        out = self.residual(torch.cat([x, fused_prompt], dim=1))

        if self.training:
            with torch.no_grad():
                global_norm = global_prompt.abs().mean().clamp_min(1e-8)
                aux: Dict[str, torch.Tensor] = {
                    'global_prompt_entropy': self._entropy(global_weights).detach(),
                    'global_prompt_abs_mean': global_prompt.abs().mean().detach(),
                    'high_frequency_gamma': self.high_frequency_scale.detach(),
                    'high_frequency_delta_prompt_abs_mean': delta_prompt.abs().mean().detach(),
                    'high_frequency_injection_ratio': (
                        self.high_frequency_scale.abs() * delta_prompt.abs().mean() / global_norm
                    ).detach(),
                    'fused_prompt_abs_mean': fused_prompt.abs().mean().detach(),
                }
                aux.update(extractor_aux)
                self.latest_aux = aux

        return out


class FrequencyGuidedSecondOrderAlignment(ModulatedDeformConv2d):
    """Frequency-guided second-order deformable alignment.

    It uses two temporal
    sources, t-1 and t-2, and predicts residual offsets/masks conditioned on
    current feature, warped features, FB residuals, previous branch feature, and
    the local high-frequency prompt-enhanced condition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 24,
        bias: bool = True,
        max_residue_magnitude: int = 10,
        prompt_dim: int = 96,
        prompt_len: int = 5,
        prompt_size: int = 96,
        mask_prior_scale: float = 0.1,
        use_hard_fb_valid: bool = False,
        ghpi_heads: int = 4,
        ghpi_interaction_scale_init: float = 0.01,
        use_high_frequency_prompt: bool = True,
        high_frequency_window_size: int = 8,
        high_frequency_zero_init: bool = True,
        frequency_mask_mode: str = 'ste_hard',
        frequency_min_half_ratio: float = 0.05,
        frequency_max_half_ratio: float = 0.45,
        frequency_mask_softness: float = 0.03,
        frequency_fixed_half_ratio: float = 0.25,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=bias,
        )

        if in_channels != 2 * out_channels:
            raise ValueError(
                'FrequencyGuidedSecondOrderAlignment expects source channels to be '
                f'2*out_channels, but got in_channels={in_channels}, '
                f'out_channels={out_channels}.'
            )
        if deform_groups % 2 != 0:
            raise ValueError('deform_groups must be even for two-source alignment.')

        self.max_residue_magnitude = float(max_residue_magnitude)
        self.mask_prior_scale = float(mask_prior_scale)
        self.use_hard_fb_valid = bool(use_hard_fb_valid)

        condition_channels = 4 * out_channels + 4
        self.condition_proj = nn.Conv2d(
            condition_channels, out_channels, 3, stride=1, padding=1, bias=True
        )
        self.ghpi = GlobalHighFrequencyPromptInteraction(
            embed_dim=out_channels,
            prompt_dim=prompt_dim,
            prompt_len=prompt_len,
            prompt_size=prompt_size,
            num_blocks=1,
            ghpi_heads=ghpi_heads,
            interaction_scale_init=ghpi_interaction_scale_init,
            use_high_frequency_prompt=use_high_frequency_prompt,
            high_frequency_window_size=high_frequency_window_size,
            high_frequency_zero_init=high_frequency_zero_init,
            frequency_mask_mode=frequency_mask_mode,
            frequency_min_half_ratio=frequency_min_half_ratio,
            frequency_max_half_ratio=frequency_max_half_ratio,
            frequency_mask_softness=frequency_mask_softness,
            frequency_fixed_half_ratio=frequency_fixed_half_ratio,
        )
        self.condition_fusion = nn.Conv2d(
            condition_channels + out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=True,
        )

        # For two-source deformable alignment:
        # offset_n1: 2 * k*k * (deform_groups/2)
        # offset_n2: 2 * k*k * (deform_groups/2)
        # mask:      k*k * deform_groups
        # These three chunks have the same channel count when deform_groups is
        # split equally between the two sources.
        kernel_area = kernel_size * kernel_size if isinstance(kernel_size, int) else kernel_size[0] * kernel_size[1]
        offset_mask_channels = deform_groups * kernel_area * 3
        self.conv_offset = nn.Sequential(
            nn.Conv2d(out_channels + 4, out_channels, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, offset_mask_channels, 3, stride=1, padding=1, bias=True),
        )
        initialize_constant(self.conv_offset[-1], 0.0, 0.0)
        self.latest_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _resize_feature(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _repeat_flow_for_offset(flow: torch.Tensor, offset_channels: int) -> torch.Tensor:
        if offset_channels % 2 != 0:
            raise ValueError('offset_channels must be even.')
        repeat = offset_channels // 2
        # Deformable offset channel order is (dy, dx), while optical flow is
        # stored as (dx, dy). Hence flow.flip(1).
        return flow.flip(1).repeat(1, repeat, 1, 1)

    def forward(
        self,
        source: torch.Tensor,
        warped_n1: torch.Tensor,
        warped_n2: torch.Tensor,
        fb_residual_n1: torch.Tensor,
        fb_residual_n2: torch.Tensor,
        previous_branch_feat: torch.Tensor,
        current_feat: torch.Tensor,
        flow_n1: torch.Tensor,
        flow_n2: torch.Tensor,
        fb_conf_n1: Optional[torch.Tensor] = None,
        fb_conf_n2: Optional[torch.Tensor] = None,
        fb_valid_n1: Optional[torch.Tensor] = None,
        fb_valid_n2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if source.ndim != 4 or current_feat.ndim != 4:
            raise ValueError('source and current_feat must be [B,C,H,W].')
        if source.size(1) != self.in_channels:
            raise ValueError(
                f'Expected source with {self.in_channels} channels, got {source.size(1)}.'
            )

        b, _, h, w = current_feat.shape
        if flow_n1.shape[-2:] != (h, w):
            flow_n1 = resize_motion(flow_n1, (h, w))
        if flow_n2.shape[-2:] != (h, w):
            flow_n2 = resize_motion(flow_n2, (h, w))

        warped_n1 = self._resize_feature(warped_n1, (h, w))
        warped_n2 = self._resize_feature(warped_n2, (h, w))
        fb_residual_n1 = self._resize_feature(fb_residual_n1, (h, w))
        fb_residual_n2 = self._resize_feature(fb_residual_n2, (h, w))
        previous_branch_feat = self._resize_feature(previous_branch_feat, (h, w))

        raw_condition = torch.cat(
            [
                warped_n1,
                warped_n2,
                fb_residual_n1,
                fb_residual_n2,
                previous_branch_feat,
                current_feat,
            ],
            dim=1,
        )
        projected_condition = self.condition_proj(raw_condition)
        prompted_condition = self.ghpi(projected_condition, current_feat)
        fused_condition = self.condition_fusion(
            torch.cat([raw_condition, prompted_condition], dim=1)
        )

        offset_mask = self.conv_offset(
            torch.cat([fused_condition, flow_n1, flow_n2], dim=1)
        )
        residual_offset_n1, residual_offset_n2, mask_logits = torch.chunk(
            offset_mask, 3, dim=1
        )

        offset_n1 = self.max_residue_magnitude * torch.tanh(residual_offset_n1)
        offset_n2 = self.max_residue_magnitude * torch.tanh(residual_offset_n2)
        offset_n1 = offset_n1 + self._repeat_flow_for_offset(
            flow_n1, offset_n1.size(1)
        )
        offset_n2 = offset_n2 + self._repeat_flow_for_offset(
            flow_n2, offset_n2.size(1)
        )
        offset = torch.cat([offset_n1, offset_n2], dim=1)

        mask_logits_n1, mask_logits_n2 = torch.chunk(mask_logits, 2, dim=1)
        if fb_conf_n1 is None:
            fb_conf_n1 = current_feat.new_ones(b, 1, h, w)
        if fb_conf_n2 is None:
            fb_conf_n2 = current_feat.new_ones(b, 1, h, w)
        if fb_valid_n1 is None:
            fb_valid_n1 = current_feat.new_ones(b, 1, h, w)
        if fb_valid_n2 is None:
            fb_valid_n2 = current_feat.new_ones(b, 1, h, w)

        fb_conf_n1 = self._resize_feature(fb_conf_n1, (h, w))
        fb_conf_n2 = self._resize_feature(fb_conf_n2, (h, w))
        fb_valid_n1 = self._resize_feature(fb_valid_n1, (h, w))
        fb_valid_n2 = self._resize_feature(fb_valid_n2, (h, w))

        mask_prior_n1 = fb_valid_n1 if self.use_hard_fb_valid else fb_conf_n1
        mask_prior_n2 = fb_valid_n2 if self.use_hard_fb_valid else fb_conf_n2
        mask_bias_n1 = self.mask_prior_scale * (2.0 * mask_prior_n1 - 1.0)
        mask_bias_n2 = self.mask_prior_scale * (2.0 * mask_prior_n2 - 1.0)
        mask_logits_n1 = mask_logits_n1 + mask_bias_n1.repeat(
            1, mask_logits_n1.size(1), 1, 1
        )
        mask_logits_n2 = mask_logits_n2 + mask_bias_n2.repeat(
            1, mask_logits_n2.size(1), 1, 1
        )
        mask = torch.sigmoid(torch.cat([mask_logits_n1, mask_logits_n2], dim=1))

        out = modulated_deform_conv2d(
            source,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )

        if self.training:
            with torch.no_grad():
                aux: Dict[str, torch.Tensor] = {
                    'deformable_offset_abs_mean': offset.abs().mean().detach(),
                    'deformable_mask_mean': mask.mean().detach(),
                    'flow_n1_abs_mean': flow_n1.abs().mean().detach(),
                    'flow_n2_abs_mean': flow_n2.abs().mean().detach(),
                    'fb_conf_n1_mean': fb_conf_n1.mean().detach(),
                    'fb_conf_n2_mean': fb_conf_n2.mean().detach(),
                    'prompted_condition_abs_mean': prompted_condition.abs().mean().detach(),
                }
                aux.update({f'ghpi_{k}': v for k, v in self.ghpi.latest_aux.items()})
                self.latest_aux = aux

        return out


class ProgressiveGatedReconstructionHead(nn.Module):
    """progressive gated reconstruction head.

    This head only replaces the reconstruction part. The frequency-guided bidirectional
    propagation and bidirectional directional temporal routing modules are kept
    unchanged.

    It follows the the compact backbone reconstruction pattern:
        trans_feat -> up21(skip_attn1(enc11)) -> decoder_level1
        -> decoder_level1_1(reverse=1) -> upsample0
        -> conv_hr0 + skip_conv(shortcut) -> out_conv -> last_conv -> + lq.

    Inputs:
        hist_feats: [B, T, C, H/4, W/4]
        lqs:        [B, T, 3, H, W]

    Output:
        restored:   [B, T, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        n_feat: int = 64,
        trans_channels: int = 192,
        kernel_size: int = 3,
        reduction: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if trans_channels <= n_feat:
            raise ValueError(
                f'trans_channels={trans_channels} must be larger than n_feat={n_feat} '
                'because SkipGuidedUpsampler expects in_channels + s_factor.'
            )

        self.in_channels = int(in_channels)
        self.n_feat = int(n_feat)
        self.trans_channels = int(trans_channels)
        self.scale_unetfeats = self.trans_channels - self.n_feat
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # progressive gated shallow reconstruction-side skip path.
        self.concat = nn.Sequential(
            nn.Conv2d(3, self.n_feat, kernel_size, 1, kernel_size // 2, bias=bias),
            self.act,
            ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act),
        )
        self.down01 = nn.Sequential(
            nn.Conv2d(self.n_feat, self.n_feat, kernel_size, 2, kernel_size // 2, bias=bias),
            self.act,
        )
        self.encoder_level1 = ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act)
        self.encoder_level1_1 = ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act)

        # Project propagated/history features to the the compact backbone transformer/reconstruction width.
        self.feat_to_trans = nn.Conv2d(self.in_channels, self.trans_channels, 3, 1, 1, bias=bias)

        # the compact backbone reconstruction modules.
        self.skip_attn1 = ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act)
        self.up21 = SkipGuidedUpsampler(self.n_feat, self.scale_unetfeats)
        self.decoder_level1 = TemporalShiftReconstructionBlock(self.n_feat, kernel_size, reduction, bias)
        self.decoder_level1_1 = TemporalShiftReconstructionBlock(self.n_feat, kernel_size, reduction, bias)
        self.upsample0 = ReconstructionUpsampler(self.n_feat, self.n_feat, 2, upsample_kernel=3)
        self.skip_conv = ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act)
        self.conv_hr0 = reconstruction_conv(self.n_feat, self.n_feat, kernel_size, bias=bias)
        self.out_conv = ReconstructionChannelBlock(self.n_feat, kernel_size, reduction, bias=bias, act=self.act)
        self.last_conv = reconstruction_conv(self.n_feat, 3, kernel_size, bias=bias)

    def _forward_one_sequence(self, feat_seq: torch.Tensor, lq_seq: torch.Tensor) -> torch.Tensor:
        """Process one video sequence.

        Args:
            feat_seq: [T, C, H/4, W/4]
            lq_seq:  [T, 3, H, W]
        """
        t, _, h, w = lq_seq.shape

        # Reconstruction-side progressive gated shallow skip features.
        shortcut = self.concat(lq_seq)
        x = self.down01(shortcut)
        enc1 = self.encoder_level1(x)
        enc11 = self.encoder_level1_1(enc1)

        # compact reconstruction from 1/4 features to full-resolution residual.
        trans_feats = self.feat_to_trans(feat_seq)
        x = self.up21(trans_feats, self.skip_attn1(enc11))
        dec1 = self.decoder_level1(x)
        dec11 = self.decoder_level1_1(dec1, reverse=1)

        dec11_up = self.upsample0(dec11)
        # Guard against odd input sizes after strided down/up sampling.
        if dec11_up.shape[-2:] != shortcut.shape[-2:]:
            dec11_up = F.interpolate(dec11_up, size=shortcut.shape[-2:], mode='bilinear', align_corners=False)

        dec11_out = self.conv_hr0(self.act(dec11_up)) + self.skip_conv(shortcut)
        dec11_out = self.out_conv(dec11_out)
        residual = self.last_conv(dec11_out)
        residual = residual[:, :, :h, :w]
        return residual + lq_seq

    def forward(self, hist_feats: torch.Tensor, lqs: torch.Tensor) -> torch.Tensor:
        if hist_feats.ndim != 5 or lqs.ndim != 5:
            raise ValueError('hist_feats and lqs must be [B,T,C,H,W].')
        b, t, c, hf, wf = hist_feats.shape
        if lqs.size(0) != b or lqs.size(1) != t:
            raise ValueError('Batch/time mismatch between hist_feats and lqs.')

        outputs: List[torch.Tensor] = []
        for batch_idx in range(b):
            outputs.append(self._forward_one_sequence(hist_feats[batch_idx], lqs[batch_idx]))
        return torch.stack(outputs, dim=0)


class ChannelGate(nn.Module):
    """Lightweight channel attention used by channel-gated fusion.

    This lightweight channel gate limits the overhead introduced by propagation
    and temporal routing:
    global average pooling -> 1x1 reduction -> activation -> 1x1 expansion ->
    sigmoid channel gate.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class ChannelGatedResidualBlock(nn.Module):
    """Convolutional Channel Attention Block for feature fusion.

    This block is used after concatenating the propagated features and the
    two directional history features. It is closer to the compact
    concat-then-channel-gated fusion than the earlier weak residual history injection.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            ChannelGate(channels, reduction=reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ChannelLayerNorm(nn.Module):
    """LayerNorm on channel dimension for BCHW feature maps."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialGatedFeedForward(nn.Module):
    """Spatially gated feed-forward network."""

    def __init__(self, dim: int, expansion: float = 1.0, bias: bool = False) -> None:
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            1,
            1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class IntermediateTemporalFusion(nn.Module):
    """Lightweight residual fusion between two alternating history layers.

    The first bidirectional temporal routing layer provides an intermediate enhancement of
    the propagated feature. No learnable residual scaling is used:

        X1 = X0 + Phi([X0, Hf1, Hb1]).
    """

    def __init__(self, channels: int, num_blocks: int = 1, reduction: int = 16) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            *[ChannelGatedResidualBlock(channels, reduction=reduction) for _ in range(max(int(num_blocks), 0))]
        )
        self.project = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(
        self,
        prop_feats: torch.Tensor,
        hist_f: torch.Tensor,
        hist_b: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.project(self.body(self.reduce(torch.cat([prop_feats, hist_f, hist_b], dim=1))))
        return prop_feats + residual


class MultiRangeTemporalFusion(nn.Module):
    """Final channel-gated fusion for X0, X1 and the second-layer bidirectional histories."""

    def __init__(self, channels: int, num_blocks: int = 2, reduction: int = 16) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = nn.Sequential(
            *[ChannelGatedResidualBlock(channels, reduction=reduction) for _ in range(max(int(num_blocks), 0))]
        )
        self.project = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        hist_f2: torch.Tensor,
        hist_b2: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x0, x1, hist_f2, hist_b2], dim=1)
        return self.project(self.body(self.reduce(x)))


class SparseHistoricalStateAlignment(nn.Module):
    """Patch-level temporal routing alignment with a full-horizon K/V cache.

    Two operations are deliberately separated:

    1. Cache update keeps every state inside ``cache_horizon``.
    2. Retrieval selects only the relative temporal offsets assigned to the
       current alternating layer.

    For a forward directional scan, offsets [1, 3, 5] select t-1, t-3 and
    t-5. During the backward scan, the same offsets select t+1, t+3 and t+5.

    Cache format:
        k_cache: [B, F, heads, N, D]
        v_cache: [B, F, heads, N, P*P*D]
    where the cache is ordered from the oldest state to the newest state in the
    current scan direction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        window_size: int = 2,
        topk: int = 5,
        local_radius: int = 2,
        cache_horizon: int = 6,
        temporal_offsets: Sequence[int] = (1, 3, 5),
        use_local_mask: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        if window_size <= 0:
            raise ValueError('window_size must be positive.')
        if cache_horizon <= 0:
            raise ValueError('cache_horizon must be positive.')

        offsets = sorted({int(v) for v in temporal_offsets})
        if not offsets or offsets[0] <= 0:
            raise ValueError(f'temporal_offsets must contain positive integers, got {temporal_offsets}.')
        if offsets[-1] > cache_horizon:
            raise ValueError(
                f'Max temporal offset {offsets[-1]} exceeds cache_horizon={cache_horizon}.'
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.window_size = int(window_size)
        self.topk = int(topk)
        self.local_radius = int(local_radius)
        self.cache_horizon = int(cache_horizon)
        self.temporal_offsets = tuple(offsets)
        self.use_local_mask = bool(use_local_mask)

        self.qk = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.q_down = nn.Conv2d(dim, dim, window_size, stride=window_size, bias=bias)
        self.k_down = nn.Conv2d(dim, dim, window_size, stride=window_size, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self._mask_cache: Dict[Tuple[int, int, int, torch.device], torch.Tensor] = {}

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        p = self.window_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def _local_mask(self, hp: int, wp: int, device: torch.device) -> torch.Tensor:
        key = (hp, wp, self.local_radius, device)
        if key in self._mask_cache:
            return self._mask_cache[key]
        y, x = torch.meshgrid(
            torch.arange(hp, device=device),
            torch.arange(wp, device=device),
            indexing='ij',
        )
        coords = torch.stack([y, x], dim=-1).view(-1, 2).float()
        dist = torch.cdist(coords, coords, p=1)
        mask = (dist <= float(self.local_radius)).view(1, 1, 1, hp * wp, hp * wp)
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _masked_softmax(attn: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        attn = attn.masked_fill(~mask, -1.0e4)
        return F.softmax(attn, dim=dim)

    def _select_temporal_cache(
        self,
        k_cached: Optional[torch.Tensor],
        v_cached: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select assigned relative offsets while preserving chronological order."""
        if k_cached is None or v_cached is None or k_cached.size(1) == 0:
            return None, None

        cache_len = k_cached.size(1)
        # Cache is oldest -> newest. Offset 1 is the newest cached state.
        # Reverse offset order so selected states remain oldest -> newest.
        indices = [cache_len - d for d in sorted(self.temporal_offsets, reverse=True) if d <= cache_len]
        if not indices:
            return None, None
        index = torch.tensor(indices, device=k_cached.device, dtype=torch.long)
        return k_cached.index_select(1, index), v_cached.index_select(1, index)

    def forward(
        self,
        x: torch.Tensor,
        k_cached: Optional[torch.Tensor] = None,
        v_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f'Expected {self.dim} channels, got {c}.')

        x_pad, pad_h, pad_w = self._pad_to_window(x)
        _, _, hpix, wpix = x_pad.shape
        p = self.window_size
        hp, wp = hpix // p, wpix // p
        num_tokens = hp * wp

        qk = self.qk_dwconv(self.qk(x_pad))
        q_map, k_map = qk.chunk(2, dim=1)
        v_map = self.v_dwconv(self.v(x_pad))

        q = self.q_down(q_map)
        k = self.k_down(k_map)
        q = rearrange(
            q,
            'b (head d) h w -> b 1 head (h w) d',
            head=self.num_heads,
            d=self.head_dim,
        )
        k = rearrange(
            k,
            'b (head d) h w -> b 1 head (h w) d',
            head=self.num_heads,
            d=self.head_dim,
        )

        v_unfold = F.unfold(v_map, kernel_size=p, stride=p)
        v_unfold = rearrange(
            v_unfold,
            'b (head d pp) n -> b 1 head n (pp d)',
            head=self.num_heads,
            d=self.head_dim,
            pp=p * p,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        cache_valid = (
            k_cached is not None
            and v_cached is not None
            and k_cached.shape[-2] == num_tokens
            and v_cached.shape[-2] == num_tokens
        )
        if not cache_valid:
            k_cached = None
            v_cached = None

        k_selected, v_selected = self._select_temporal_cache(k_cached, v_cached)
        if k_selected is not None and v_selected is not None:
            k_retrieval = torch.cat([k_selected, k], dim=1)
            v_retrieval = torch.cat([v_selected, v_unfold], dim=1)
        else:
            k_retrieval = k
            v_retrieval = v_unfold

        # Update the full candidate cache independently from this layer's
        # retrieval subset. This is essential for alternating offsets.
        if k_cached is not None and v_cached is not None:
            k_updated = torch.cat([k_cached, k], dim=1)
            v_updated = torch.cat([v_cached, v_unfold], dim=1)
        else:
            k_updated = k
            v_updated = v_unfold

        attn = torch.matmul(q, k_retrieval.transpose(-2, -1)) * self.temperature
        k_top = min(max(self.topk, 1), attn.size(-1))
        _, topk_idx = torch.topk(attn, k=k_top, dim=-1)
        topk_mask = torch.zeros_like(attn, dtype=torch.bool)
        topk_mask.scatter_(dim=-1, index=topk_idx, value=True)

        if self.use_local_mask:
            local_mask = self._local_mask(hp, wp, x.device)
            sparse_mask = topk_mask | local_mask
        else:
            sparse_mask = topk_mask
        attn = self._masked_softmax(attn, sparse_mask, dim=-1)

        out = torch.matmul(attn, v_retrieval)
        f_retrieval = out.size(1)
        out = rearrange(out, 'b f head n ppd -> (b f) (head ppd) n')
        out = F.fold(out, output_size=(hpix, wpix), kernel_size=p, stride=p)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        out = self.project_out(out)
        out = rearrange(out, '(b f) c h w -> b f c h w', b=b, f=f_retrieval)

        k_to_cache = k_updated[:, -self.cache_horizon :, ...].detach()
        v_to_cache = v_updated[:, -self.cache_horizon :, ...].detach()
        return out, k_to_cache, v_to_cache


class ResidualConsistentStateSelector(nn.Module):
    """Patch-wise hard selection over aligned historical states.

    ``SparseHistoricalStateAlignment`` first performs the original patch-level
    patch-level sparse matching independently for every selected cached frame.
    Its output is ordered as::

        [aligned history state 1, ..., aligned history state F,
         current self-reference]

    This module compares every aligned historical state with the current
    self-reference in a normalized feature domain. The residual is pooled using
    the same patch size as the corresponding history layer. At every patch
    location, only the ``topm`` historical states with the smallest residuals
    are gathered. The gathered patches are rebuilt into ``topm`` rank-ordered
    history slots, followed by the unchanged current self-reference.

    Unlike merely multiplying rejected states by zero, gathering physically
    reduces the number of states passed to the K/V projection and
    ``TemporalFeatureRouter``. The first sparse-attention matmul in
    ``SparseHistoricalStateAlignment`` is still required because the residual can only be
    measured after alignment.
    """

    def __init__(
        self,
        window_size: int,
        topm: int,
        normalize_features: bool = True,
        detach_selection: bool = True,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError('window_size must be positive.')
        if topm < 0:
            raise ValueError('topm must be non-negative.')
        self.window_size = int(window_size)
        self.topm = int(topm)
        self.normalize_features = bool(normalize_features)
        self.detach_selection = bool(detach_selection)

    def forward(
        self,
        x_spatial: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select residual-consistent history states patch by patch.

        Args:
            x_spatial: [B, F_history + 1, C, H, W]. The last state is the
                current self-reference.

        Returns:
            sparse_states: [B, min(topm,F_history)+1, C, H, W] when hard
                selection is active. The final state is always the current
                self-reference.
            selected_idx: [B, topm, Hp, Wp] source-frame indices, or ``None``
                when selection is bypassed.
            patch_residual: [B, F_history, 1, Hp, Wp], or ``None``.
        """
        if x_spatial.ndim != 5:
            raise ValueError(
                f'x_spatial must be [B,F,C,H,W], got {tuple(x_spatial.shape)}.'
            )
        if x_spatial.size(1) <= 1:
            return x_spatial, None, None

        aligned_hist = x_spatial[:, :-1]
        current_ref = x_spatial[:, -1:]
        b, num_hist, c, h, w = aligned_hist.shape

        # topm == 0 disables hard selection for clean ablation. If the number
        # of available history states is already no larger than topm, retaining
        # all states avoids artificial duplication near sequence boundaries.
        keep_num = min(self.topm, num_hist)
        if keep_num <= 0 or keep_num >= num_hist:
            return x_spatial, None, None

        if self.normalize_features:
            hist_cmp = F.normalize(aligned_hist, dim=2, eps=1e-6)
            cur_cmp = F.normalize(current_ref, dim=2, eps=1e-6)
        else:
            hist_cmp = aligned_hist
            cur_cmp = current_ref

        residual = torch.mean(
            torch.abs(hist_cmp - cur_cmp),
            dim=2,
            keepdim=True,
        )
        # residual: [B, F_history, 1, H, W]

        residual_for_select = residual.detach() if self.detach_selection else residual
        patch_residual = F.avg_pool2d(
            residual_for_select.reshape(b * num_hist, 1, h, w),
            kernel_size=self.window_size,
            stride=self.window_size,
            ceil_mode=True,
            count_include_pad=False,
        )
        hp, wp = patch_residual.shape[-2:]
        patch_residual = patch_residual.view(b, num_hist, 1, hp, wp)

        # sorted=True gives the output slots a consistent semantic meaning:
        # slot 0 is the most residual-consistent state, slot 1 the second, etc.
        selected_idx = torch.topk(
            patch_residual.squeeze(2),
            k=keep_num,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        # selected_idx: [B, keep_num, Hp, Wp]

        p = self.window_size
        h_pad = hp * p
        w_pad = wp * p
        pad_h = h_pad - h
        pad_w = w_pad - w

        hist_flat = aligned_hist.reshape(b * num_hist, c, h, w)
        if pad_h > 0 or pad_w > 0:
            hist_flat = F.pad(hist_flat, (0, pad_w, 0, pad_h))
        hist_pad = hist_flat.view(b, num_hist, c, h_pad, w_pad)

        # [B,F,C,Hp,p,Wp,p] -> [B,F,Hp,Wp,C,p,p]
        hist_patches = hist_pad.view(b, num_hist, c, hp, p, wp, p)
        hist_patches = hist_patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous()

        gather_idx = selected_idx[..., None, None, None].expand(
            -1, -1, -1, -1, c, p, p
        )
        selected_patches = torch.gather(
            hist_patches,
            dim=1,
            index=gather_idx,
        )
        # [B,K,Hp,Wp,C,p,p] -> [B,K,C,Hpad,Wpad]
        selected_hist = selected_patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        selected_hist = selected_hist.view(b, keep_num, c, h_pad, w_pad)
        selected_hist = selected_hist[:, :, :, :h, :w]

        sparse_states = torch.cat([selected_hist, current_ref], dim=1)
        return sparse_states, selected_idx, patch_residual


class TemporalFeatureRouter(nn.Module):
    """patch-level channel router over aligned historical states."""

    def __init__(self, dim: int, num_heads: int = 1, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        k_hist: Optional[torch.Tensor] = None,
        v_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k_cur, v_cur = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        k_cur = rearrange(k_cur, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        v_cur = rearrange(v_cur, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k_cur = F.normalize(k_cur, dim=-1)

        if k_hist is not None and v_hist is not None:
            k = torch.cat([k_hist, k_cur], dim=2)
            v = torch.cat([v_hist, v_cur], dim=2)
        else:
            k, v = k_cur, v_cur

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head d (h w) -> b (head d) h w', h=h, w=w)
        return self.project_out(out)


class DirectionalTemporalRoutingLayer(nn.Module):
    """Directional temporal routing over an assigned temporal-offset subset."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        window_size: int = 2,
        topk: int = 5,
        local_radius: int = 2,
        cache_horizon: int = 6,
        temporal_offsets: Sequence[int] = (1, 3, 5),
        use_local_mask: bool = True,
        use_residual_selection: bool = True,
        residual_topm: int = 2,
        residual_normalize_features: bool = True,
        residual_detach_selection: bool = True,
        ffn_expansion: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm(dim)
        self.state_alignment = SparseHistoricalStateAlignment(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            topk=topk,
            local_radius=local_radius,
            cache_horizon=cache_horizon,
            temporal_offsets=temporal_offsets,
            use_local_mask=use_local_mask,
            bias=bias,
        )
        self.use_residual_selection = bool(use_residual_selection)
        self.residual_selector = ResidualConsistentStateSelector(
            window_size=window_size,
            topm=residual_topm,
            normalize_features=residual_normalize_features,
            detach_selection=residual_detach_selection,
        )
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.temporal_router = TemporalFeatureRouter(dim, num_heads=num_heads, bias=bias)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = SpatialGatedFeedForward(dim, expansion=ffn_expansion, bias=bias)
        self.num_heads = int(num_heads)

    def forward(
        self,
        x: torch.Tensor,
        k_cached: Optional[torch.Tensor] = None,
        v_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        x_spatial, k_to_cache, v_to_cache = self.state_alignment(x_norm, k_cached, v_cached)

        if self.use_residual_selection:
            x_spatial, _, _ = self.residual_selector(x_spatial)

        b, f, c, h, w = x_spatial.shape
        x_hist = rearrange(x_spatial, 'b f c h w -> (b f) c h w')
        kv = self.kv_dwconv(self.kv(x_hist))
        k_hist, v_hist = kv.chunk(2, dim=1)
        k_hist = rearrange(
            k_hist,
            '(b f) (head d) h w -> b head (f d) (h w)',
            b=b,
            f=f,
            head=self.num_heads,
        )
        v_hist = rearrange(
            v_hist,
            '(b f) (head d) h w -> b head (f d) (h w)',
            b=b,
            f=f,
            head=self.num_heads,
        )
        k_hist = F.normalize(k_hist, dim=-1)

        out = self.temporal_router(x_norm, k_hist, v_hist)
        out = x + out
        out = out + self.ffn(self.norm2(out))
        return out, k_to_cache, v_to_cache


class BidirectionalMultiRangeTemporalRouting(nn.Module):
    """Two-stage bidirectional multi-range temporal routing.

    Layer 1 and layer 2 can use independent patch-window sizes, attention
    top-k values, local-radius settings, and residual-consistency selection top-m
    values. By default, layer 1 keeps local+global matching for near history,
    while layer 2 disables the local-radius mask and uses coarser global
    matching for long-range history. The two layers independently rebuild their
    K/V caches from X0 and X1.

    Pipeline:
        X0 = propagated features
        Hf1, Hb1 = BMTR_1(X0; near-range offsets)
        X1 = X0 + inter-layer fusion([X0,Hf1,Hb1])
        Hf2, Hb2 = BMTR_2(X1; long-range offsets)
        Xout = final fusion([X0,X1,Hf2,Hb2])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        layer_window_size: Sequence[int] = (2, 4),
        layer_topk: Sequence[int] = (5, 3),
        cache_horizon: int = 12,
        layer_offsets: Sequence[Sequence[int]] = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),
        layer_local_radius: Sequence[int] = (2, 0),
        layer_use_local: Sequence[bool] = (True, False),
        layer_use_residual_selection: Sequence[bool] = (True, True),
        layer_residual_topm: Sequence[int] = (3, 2),
        layer_residual_normalize_features: Sequence[bool] = (True, True),
        residual_detach_selection: bool = True,
        ffn_expansion: float = 1.0,
        bias: bool = False,
        intermediate_fusion_blocks: int = 1,
        final_fusion_blocks: int = 2,
        channel_reduction: int = 16,
    ) -> None:
        super().__init__()
        if len(layer_offsets) != 2:
            raise ValueError(f'Exactly two alternating layers are required, got {len(layer_offsets)}.')
        if (
            len(layer_window_size) != 2
            or len(layer_topk) != 2
            or len(layer_local_radius) != 2
            or len(layer_use_local) != 2
            or len(layer_use_residual_selection) != 2
            or len(layer_residual_topm) != 2
            or len(layer_residual_normalize_features) != 2
        ):
            raise ValueError(
                'All layer-wise history settings must contain exactly two values.'
            )

        window_sizes = [int(v) for v in layer_window_size]
        topks = [int(v) for v in layer_topk]
        if any(v <= 0 for v in window_sizes):
            raise ValueError(f'All layer window sizes must be positive, got {window_sizes}.')
        if any(v <= 0 for v in topks):
            raise ValueError(f'All layer top-k values must be positive, got {topks}.')
        residual_topms = [int(v) for v in layer_residual_topm]
        if any(v < 0 for v in residual_topms):
            raise ValueError(
                f'All residual-consistency top-m values must be non-negative, got {residual_topms}.'
            )

        offsets = [tuple(int(v) for v in layer) for layer in layer_offsets]
        max_offset = max(max(layer) for layer in offsets)
        if cache_horizon < max_offset:
            raise ValueError(
                f'cache_horizon={cache_horizon} is smaller than max configured offset={max_offset}.'
            )

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        for layer_idx in range(2):
            kwargs = dict(
                dim=dim,
                num_heads=num_heads,
                window_size=window_sizes[layer_idx],
                topk=topks[layer_idx],
                local_radius=int(layer_local_radius[layer_idx]),
                cache_horizon=cache_horizon,
                temporal_offsets=offsets[layer_idx],
                use_local_mask=bool(layer_use_local[layer_idx]),
                use_residual_selection=bool(layer_use_residual_selection[layer_idx]),
                residual_topm=residual_topms[layer_idx],
                residual_normalize_features=bool(
                    layer_residual_normalize_features[layer_idx]
                ),
                residual_detach_selection=bool(residual_detach_selection),
                ffn_expansion=ffn_expansion,
                bias=bias,
            )
            self.forward_layers.append(DirectionalTemporalRoutingLayer(**kwargs))
            self.backward_layers.append(DirectionalTemporalRoutingLayer(**kwargs))

        self.inter_layer_fusion = IntermediateTemporalFusion(
            channels=dim,
            num_blocks=intermediate_fusion_blocks,
            reduction=channel_reduction,
        )
        self.final_fusion = MultiRangeTemporalFusion(
            channels=dim,
            num_blocks=final_fusion_blocks,
            reduction=channel_reduction,
        )
        self.layer_offsets = offsets
        self.layer_window_size = tuple(window_sizes)
        self.layer_topk = tuple(topks)
        self.layer_residual_topm = tuple(residual_topms)
        self.cache_horizon = int(cache_horizon)

    def _run_direction(
        self,
        feats: torch.Tensor,
        layer_idx: int,
        reverse: bool,
    ) -> torch.Tensor:
        _, t, _, _, _ = feats.shape
        module = self.backward_layers[layer_idx] if reverse else self.forward_layers[layer_idx]
        indices = range(t - 1, -1, -1) if reverse else range(t)
        k_cache: Optional[torch.Tensor] = None
        v_cache: Optional[torch.Tensor] = None
        outs: List[torch.Tensor] = []

        for idx in indices:
            out, k_cache, v_cache = module(feats[:, idx], k_cache, v_cache)
            outs.append(out)
        if reverse:
            outs = outs[::-1]
        return torch.stack(outs, dim=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 5:
            raise ValueError(f'Expected [B,T,C,H,W], got {tuple(feats.shape)}.')
        b, t, c, h, w = feats.shape
        x0 = feats

        hist_f1 = self._run_direction(x0, layer_idx=0, reverse=False)
        hist_b1 = self._run_direction(x0, layer_idx=0, reverse=True)
        x1 = self.inter_layer_fusion(
            prop_feats=x0.reshape(b * t, c, h, w),
            hist_f=hist_f1.reshape(b * t, c, h, w),
            hist_b=hist_b1.reshape(b * t, c, h, w),
        ).view(b, t, c, h, w)

        # Layer 2 builds a fresh cache from X1 instead of reusing layer-1 K/V.
        hist_f2 = self._run_direction(x1, layer_idx=1, reverse=False)
        hist_b2 = self._run_direction(x1, layer_idx=1, reverse=True)

        fused = self.final_fusion(
            x0=x0.reshape(b * t, c, h, w),
            x1=x1.reshape(b * t, c, h, w),
            hist_f2=hist_f2.reshape(b * t, c, h, w),
            hist_b2=hist_b2.reshape(b * t, c, h, w),
        )
        return fused.view(b, t, c, h, w)


class FrequencyGuidedSecondOrderPropagation(nn.Module):
    """Frequency-guided bidirectional second-order propagation.

    It performs only:
        backward -> forward
    The module performs one backward pass followed by one forward pass. Each
    direction uses ``FrequencyGuidedSecondOrderAlignment`` with GHPI-enhanced
    alignment conditions.
    """

    def __init__(
        self,
        mid_channels: int = 96,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        deform_groups: int = 24,
        prompt_dim: int = 96,
        prompt_len: int = 5,
        prompt_size: int = 96,
        mask_prior_scale: float = 0.1,
        use_hard_fb_valid: bool = False,
        ghpi_heads: int = 4,
        ghpi_interaction_scale_init: float = 0.01,
        use_high_frequency_prompt: bool = True,
        high_frequency_window_size: int = 8,
        high_frequency_zero_init: bool = True,
        frequency_mask_mode: str = 'ste_hard',
        frequency_min_half_ratio: float = 0.05,
        frequency_max_half_ratio: float = 0.45,
        frequency_mask_softness: float = 0.03,
        frequency_fixed_half_ratio: float = 0.25,
        fb_alpha1: float = 0.01,
        fb_alpha2: float = 0.5,
    ) -> None:
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.fb_alpha1 = float(fb_alpha1)
        self.fb_alpha2 = float(fb_alpha2)

        align_kwargs = dict(
            in_channels=2 * mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            deform_groups=deform_groups,
            bias=True,
            max_residue_magnitude=max_residue_magnitude,
            prompt_dim=prompt_dim,
            prompt_len=prompt_len,
            prompt_size=prompt_size,
            mask_prior_scale=mask_prior_scale,
            use_hard_fb_valid=use_hard_fb_valid,
            ghpi_heads=ghpi_heads,
            ghpi_interaction_scale_init=ghpi_interaction_scale_init,
            use_high_frequency_prompt=use_high_frequency_prompt,
            high_frequency_window_size=high_frequency_window_size,
            high_frequency_zero_init=high_frequency_zero_init,
            frequency_mask_mode=frequency_mask_mode,
            frequency_min_half_ratio=frequency_min_half_ratio,
            frequency_max_half_ratio=frequency_max_half_ratio,
            frequency_mask_softness=frequency_mask_softness,
            frequency_fixed_half_ratio=frequency_fixed_half_ratio,
        )
        self.backward_align = FrequencyGuidedSecondOrderAlignment(**align_kwargs)
        self.forward_align = FrequencyGuidedSecondOrderAlignment(**align_kwargs)

        self.backward_refine = ResidualBlocksWithInputConv(
            mid_channels * 3, mid_channels, num_blocks
        )
        self.forward_refine = ResidualBlocksWithInputConv(
            mid_channels * 3, mid_channels, num_blocks
        )
        self.prop_fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, max(2, num_blocks // 2)),
        )

    def _align_one_step(
        self,
        align: FrequencyGuidedSecondOrderAlignment,
        feat_curr: torch.Tensor,
        feat_prev: torch.Tensor,
        feat_prev2: Optional[torch.Tensor],
        prev_branch_feat: torch.Tensor,
        flow_n1: torch.Tensor,
        rev_flow_n1: torch.Tensor,
        prev_flow_n1: Optional[torch.Tensor],
        prev_rev_flow_n1: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # First-order source and condition.
        cond_n1 = warp_feature(feat_prev, flow_n1)
        fb_res_n1, fb_conf_n1, fb_valid_n1 = bidirectional_motion_consistency(
            flow_n1, rev_flow_n1, alpha1=self.fb_alpha1, alpha2=self.fb_alpha2
        )

        # Second-order source and condition.
        if feat_prev2 is not None and prev_flow_n1 is not None and prev_rev_flow_n1 is not None:
            flow_n2 = compose_motion(flow_n1, prev_flow_n1)
            rev_flow_n2 = compose_motion(prev_rev_flow_n1, rev_flow_n1)
            cond_n2 = warp_feature(feat_prev2, flow_n2)
            fb_res_n2, fb_conf_n2, fb_valid_n2 = bidirectional_motion_consistency(
                flow_n2, rev_flow_n2, alpha1=self.fb_alpha1, alpha2=self.fb_alpha2
            )
            source_n2 = feat_prev2
        else:
            flow_n2 = torch.zeros_like(flow_n1)
            cond_n2 = torch.zeros_like(cond_n1)
            fb_res_n2 = torch.zeros_like(fb_res_n1)
            fb_conf_n2 = torch.zeros_like(fb_conf_n1)
            fb_valid_n2 = torch.zeros_like(fb_valid_n1)
            source_n2 = torch.zeros_like(feat_prev)

        source = torch.cat([feat_prev, source_n2], dim=1)
        feat_align = align(
            source=source,
            warped_n1=cond_n1,
            warped_n2=cond_n2,
            fb_residual_n1=fb_res_n1,
            fb_residual_n2=fb_res_n2,
            previous_branch_feat=prev_branch_feat,
            current_feat=feat_curr,
            flow_n1=flow_n1,
            flow_n2=flow_n2,
            fb_conf_n1=fb_conf_n1,
            fb_conf_n2=fb_conf_n2,
            fb_valid_n1=fb_valid_n1,
            fb_valid_n2=fb_valid_n2,
        )
        return feat_align, flow_n1.detach(), rev_flow_n1.detach(), fb_valid_n1.detach()

    def _propagate_direction(
        self,
        spatial_feats: torch.Tensor,
        flows: torch.Tensor,
        rev_flows: torch.Tensor,
        direction: str,
        prev_branch_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, c, h, w = spatial_feats.shape
        assert direction in {'backward', 'forward'}
        align = self.backward_align if direction == 'backward' else self.forward_align
        refine = self.backward_refine if direction == 'backward' else self.forward_refine
        indices = range(t - 1, -1, -1) if direction == 'backward' else range(t)

        feat_prop = spatial_feats.new_zeros(b, c, h, w)
        history_feats: List[torch.Tensor] = []
        outs: List[torch.Tensor] = []
        prev_flow_n1: Optional[torch.Tensor] = None
        prev_rev_flow_n1: Optional[torch.Tensor] = None

        for step, idx in enumerate(indices):
            feat_curr = spatial_feats[:, idx]
            if prev_branch_feats is None:
                prev_branch = torch.zeros_like(feat_curr)
            else:
                prev_branch = prev_branch_feats[:, idx]

            if step == 0:
                feat_align = torch.zeros_like(feat_curr)
                prev_flow_n1 = None
                prev_rev_flow_n1 = None
            else:
                if direction == 'backward':
                    flow_idx = idx  # current idx -> idx+1
                else:
                    flow_idx = idx - 1  # current idx -> idx-1

                flow_n1 = flows[:, flow_idx]
                rev_flow_n1 = rev_flows[:, flow_idx]
                feat_prev = history_feats[-1]
                feat_prev2 = history_feats[-2] if step > 1 else None
                feat_align, prev_flow_n1, prev_rev_flow_n1, _ = self._align_one_step(
                    align=align,
                    feat_curr=feat_curr,
                    feat_prev=feat_prev,
                    feat_prev2=feat_prev2,
                    prev_branch_feat=prev_branch,
                    flow_n1=flow_n1,
                    rev_flow_n1=rev_flow_n1,
                    prev_flow_n1=prev_flow_n1,
                    prev_rev_flow_n1=prev_rev_flow_n1,
                )

            refine_in = torch.cat([feat_curr, prev_branch, feat_align], dim=1)
            feat_prop = feat_align + refine(refine_in)
            history_feats.append(feat_prop)
            outs.append(feat_prop)

        if direction == 'backward':
            outs = outs[::-1]
        return torch.stack(outs, dim=1)

    def forward(
        self,
        spatial_feats: torch.Tensor,
        flows_forward: torch.Tensor,
        flows_backward: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # backward: frame i queries source i+1, hence flows_backward.
        feats_backward = self._propagate_direction(
            spatial_feats=spatial_feats,
            flows=flows_backward,
            rev_flows=flows_forward,
            direction='backward',
            prev_branch_feats=None,
        )
        # forward: frame i queries source i-1, hence flows_forward.
        feats_forward = self._propagate_direction(
            spatial_feats=spatial_feats,
            flows=flows_forward,
            rev_flows=flows_backward,
            direction='forward',
            prev_branch_feats=feats_backward,
        )
        b, t, c, h, w = spatial_feats.shape
        prop = self.prop_fusion(
            torch.cat([spatial_feats, feats_backward, feats_forward], dim=2).view(b * t, c * 3, h, w)
        ).view(b, t, c, h, w)
        return prop, feats_backward, feats_forward


@ARCH_REGISTRY.register()
class FrequencyGuidedMultiRangeRestorationNet(nn.Module):
    """Frequency-guided propagation with bidirectional multi-range temporal routing.

    Expected input:
        lqs: [B, T, 3, H, W]

    Output:
        restored: [B, T, 3, H, W]
    """

    def __init__(
        self,
        mid_channels: int = 96,
        num_extract_blocks: int = 5,
        num_propagation_blocks: int = 7,
        max_residue_magnitude: int = 10,
        motion_estimator_pretrained: Optional[str] = None,
        freeze_motion_estimator: bool = True,
        deform_groups: int = 24,
        prompt_size: int = 96,
        prompt_dim: int = 96,
        prompt_len: int = 5,
        mask_prior_scale: float = 0.1,
        use_hard_fb_valid: bool = False,
        fb_alpha1: float = 0.01,
        fb_alpha2: float = 0.5,
        ghpi_heads: int = 4,
        ghpi_interaction_scale_init: float = 0.01,
        use_high_frequency_prompt: bool = True,
        high_frequency_window_size: int = 8,
        high_frequency_zero_init: bool = True,
        frequency_mask_mode: str = 'ste_hard',
        frequency_min_half_ratio: float = 0.05,
        frequency_max_half_ratio: float = 0.45,
        frequency_mask_softness: float = 0.03,
        frequency_fixed_half_ratio: float = 0.25,
        routing_num_heads: int = 1,
        routing_layer_window_size: Sequence[int] = (2, 4),
        routing_layer_topk: Sequence[int] = (5, 3),
        routing_cache_horizon: int = 12,
        routing_layer_offsets: Sequence[Sequence[int]] = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)),
        routing_layer_local_radius: Sequence[int] = (2, 0),
        routing_layer_use_local: Sequence[bool] = (True, False),
        routing_layer_use_residual_selection: Sequence[bool] = (True, True),
        routing_layer_residual_topm: Sequence[int] = (3, 2),
        routing_layer_normalize_residual: Sequence[bool] = (True, True),
        routing_detach_selection: bool = True,
        routing_ffn_expansion: float = 1.0,
        routing_intermediate_fusion_blocks: int = 1,
        routing_final_fusion_blocks: int = 2,
        routing_channel_reduction: int = 16,
        use_temporal_routing: bool = True,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.freeze_motion_estimator = bool(freeze_motion_estimator)
        self.use_temporal_routing = bool(use_temporal_routing)
        self.use_checkpoint = bool(use_checkpoint)

        self.motion_estimator = MotionEstimator(motion_estimator_pretrained)
        if self.freeze_motion_estimator:
            for p in self.motion_estimator.parameters():
                p.requires_grad_(False)

        # Quarter-resolution spatial feature extraction.
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, num_extract_blocks),
        )

        self.propagation = FrequencyGuidedSecondOrderPropagation(
            mid_channels=mid_channels,
            num_blocks=num_propagation_blocks,
            max_residue_magnitude=max_residue_magnitude,
            deform_groups=deform_groups,
            prompt_dim=prompt_dim,
            prompt_len=prompt_len,
            prompt_size=prompt_size,
            mask_prior_scale=mask_prior_scale,
            use_hard_fb_valid=use_hard_fb_valid,
            ghpi_heads=ghpi_heads,
            ghpi_interaction_scale_init=ghpi_interaction_scale_init,
            use_high_frequency_prompt=use_high_frequency_prompt,
            high_frequency_window_size=high_frequency_window_size,
            high_frequency_zero_init=high_frequency_zero_init,
            frequency_mask_mode=frequency_mask_mode,
            frequency_min_half_ratio=frequency_min_half_ratio,
            frequency_max_half_ratio=frequency_max_half_ratio,
            frequency_mask_softness=frequency_mask_softness,
            frequency_fixed_half_ratio=frequency_fixed_half_ratio,
            fb_alpha1=fb_alpha1,
            fb_alpha2=fb_alpha2,
        )

        if self.use_temporal_routing:
            self.temporal_routing = BidirectionalMultiRangeTemporalRouting(
                dim=mid_channels,
                num_heads=routing_num_heads,
                layer_window_size=routing_layer_window_size,
                layer_topk=routing_layer_topk,
                cache_horizon=routing_cache_horizon,
                layer_offsets=routing_layer_offsets,
                layer_local_radius=routing_layer_local_radius,
                layer_use_local=routing_layer_use_local,
                layer_use_residual_selection=routing_layer_use_residual_selection,
                layer_residual_topm=routing_layer_residual_topm,
                layer_residual_normalize_features=(
                    routing_layer_normalize_residual
                ),
                residual_detach_selection=routing_detach_selection,
                ffn_expansion=routing_ffn_expansion,
                bias=False,
                intermediate_fusion_blocks=routing_intermediate_fusion_blocks,
                final_fusion_blocks=routing_final_fusion_blocks,
                channel_reduction=routing_channel_reduction,
            )
        else:
            self.temporal_routing = nn.Identity()

        # Progressive gated reconstruction head.
        self.reconstruction = ProgressiveGatedReconstructionHead(
            in_channels=mid_channels,
            n_feat=32,
            trans_channels=128,
            kernel_size=3,
            reduction=8,
            bias=False,
        )

        self.latest_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _translate_legacy_checkpoint_key(key: str) -> str:
        """Translate state-dict keys produced by the earlier experimental code.

        The translation is intentionally limited to renamed modules. It does not
        hide or rewrite third-party dependencies and does not alter tensor values.
        """
        replacements = (
            ('spynet.', 'motion_estimator.'),
            ('.PGI_align.', '.ghpi.'),
            ('.hf_window_fusion.', '.local_prompt_interaction.'),
            ('.gamma_hf', '.high_frequency_scale'),
            (
                '.high_frequency_extractor.mgb.',
                '.high_frequency_extractor.frequency_mask_generator.',
            ),
            ('sparse_history.', 'temporal_routing.'),
            ('.spatial_aligner.', '.state_alignment.'),
            ('.residual_sparsifier.', '.residual_selector.'),
            ('.router.', '.temporal_router.'),
        )
        translated = key
        for old, new in replacements:
            translated = translated.replace(old, new)
        return translated

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load both paper-named and earlier experimental checkpoints."""
        translated = OrderedDict()
        for key, value in state_dict.items():
            new_key = self._translate_legacy_checkpoint_key(key)
            if new_key in translated and new_key != key:
                raise KeyError(
                    f'Checkpoint key collision after translation: {key!r} -> {new_key!r}.'
                )
            translated[new_key] = value
        if hasattr(state_dict, '_metadata'):
            translated._metadata = state_dict._metadata
        return super().load_state_dict(translated, strict=strict)

    def estimate_bidirectional_motion(self, lqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute full-resolution bidirectional adjacent flows.

        flows_forward[:, i] : frame i+1 -> frame i
        flows_backward[:, i]: frame i   -> frame i+1
        Both are target-to-source flows for BasicSR ``flow_warp`` usage.
        """
        b, t, c, h, w = lqs.shape
        if t <= 1:
            empty = lqs.new_zeros(b, 0, 2, h, w)
            return empty, empty

        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        ctx = torch.no_grad() if self.freeze_motion_estimator else nullcontext()
        with ctx:
            flows_backward = self.motion_estimator(lqs_1, lqs_2).view(b, t - 1, 2, h, w)
            flows_forward = self.motion_estimator(lqs_2, lqs_1).view(b, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def extract_spatial_features(self, lqs: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = lqs.shape
        feats = self.feat_extract(lqs.view(b * t, c, h, w))
        _, c_feat, hf, wf = feats.shape
        return feats.view(b, t, c_feat, hf, wf)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        if lqs.ndim != 5:
            raise ValueError(f'Expected lqs [B,T,C,H,W], got {tuple(lqs.shape)}.')
        b, t, c, h, w = lqs.shape
        if c != 3:
            raise ValueError(f'Expected RGB input with 3 channels, got {c}.')

        spatial_feats = self.extract_spatial_features(lqs)
        hf, wf = spatial_feats.shape[-2:]

        flows_forward, flows_backward = self.estimate_bidirectional_motion(lqs)
        flows_forward = resize_motion_sequence(flows_forward, (hf, wf))
        flows_backward = resize_motion_sequence(flows_backward, (hf, wf))

        prop_feats, feats_backward, feats_forward = self.propagation(
            spatial_feats, flows_forward, flows_backward
        )

        if self.use_temporal_routing:
            if self.training and self.use_checkpoint:
                hist_feats = checkpoint(
                    self.temporal_routing,
                    prop_feats,
                    use_reentrant=False,
                )
            else:
                hist_feats = self.temporal_routing(prop_feats)
        else:
            hist_feats = prop_feats

        if self.training and self.use_checkpoint:
            out = checkpoint(
                self.reconstruction,
                hist_feats,
                lqs,
                use_reentrant=False,
            )
        else:
            out = self.reconstruction(hist_feats, lqs)
        out = out[:, :, :, :h, :w]

        if self.training:
            with torch.no_grad():
                self.latest_aux = {
                    'prop_abs_mean': prop_feats.abs().mean().detach(),
                    'routed_abs_mean': hist_feats.abs().mean().detach(),
                    'backward_abs_mean': feats_backward.abs().mean().detach(),
                    'forward_abs_mean': feats_forward.abs().mean().detach(),
                }
        return out
