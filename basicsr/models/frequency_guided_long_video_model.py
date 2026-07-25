"""Memory-bounded temporal-chunk and spatial-tile video inference."""

import math

import torch
import torch.nn.functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .frequency_guided_video_restoration_model import (
    FrequencyGuidedVideoRestorationModel,
)


@MODEL_REGISTRY.register()
class FrequencyGuidedLongVideoModel(FrequencyGuidedVideoRestorationModel):
    """Inference wrapper that keeps complete long sequences in CPU memory."""

    def feed_data(self, data):
        if self.is_train:
            return super().feed_data(data)
        self.lq = data['lq'].detach().cpu()
        if 'gt' in data:
            self.gt = data['gt'].detach().cpu()

    @staticmethod
    def _tile_starts(length, tile_size):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, tile_size))
        last_start = length - tile_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _pad_spatial(tile, minimum_size, size_multiple):
        height, width = tile.shape[-2:]
        target_height = max(minimum_size, math.ceil(height / size_multiple) * size_multiple)
        target_width = max(minimum_size, math.ceil(width / size_multiple) * size_multiple)
        pad_height = target_height - height
        pad_width = target_width - width
        if not pad_height and not pad_width:
            return tile, height, width
        mode = (
            'reflect'
            if height > pad_height and width > pad_width
            else 'replicate'
        )
        batch, frames, channels = tile.shape[:3]
        tile = tile.reshape(batch * frames, channels, height, width)
        tile = F.pad(
            tile,
            (0, pad_width, 0, pad_height),
            mode=mode,
        )
        tile = tile.reshape(
            batch,
            frames,
            channels,
            target_height,
            target_width,
        )
        return tile, height, width

    @torch.no_grad()
    def tile_forward_video(
        self,
        net,
        lq,
        tile_size=256,
        tile_pad=32,
        scale=1,
        minimum_size=64,
        size_multiple=4,
    ):
        """Restore one CUDA/device-resident temporal chunk by spatial tiles."""
        if lq.dim() != 5:
            raise ValueError(f'Expected [B,T,C,H,W], got {tuple(lq.shape)}.')
        batch, frames, _, height, width = lq.shape
        if batch != 1:
            raise ValueError('Long-video tiled inference requires batch size 1.')
        if tile_size <= 0 or tile_pad < 0:
            raise ValueError('tile_size must be positive and tile_pad non-negative.')
        if scale < 1 or minimum_size < 1 or size_multiple < 1:
            raise ValueError('scale, minimum_size, and size_multiple must be positive.')

        output = None
        weight = None
        for top in self._tile_starts(height, tile_size):
            for left in self._tile_starts(width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                padded_top = max(top - tile_pad, 0)
                padded_left = max(left - tile_pad, 0)
                padded_bottom = min(bottom + tile_pad, height)
                padded_right = min(right + tile_pad, width)

                tile = lq[
                    :, :, :,
                    padded_top:padded_bottom,
                    padded_left:padded_right,
                ]
                tile, original_height, original_width = self._pad_spatial(
                    tile,
                    minimum_size=minimum_size,
                    size_multiple=size_multiple,
                )
                tile_output = self._unwrap_output(net(tile))
                tile_output = tile_output[
                    :, :, :,
                    :original_height * scale,
                    :original_width * scale,
                ]

                if output is None:
                    output = tile_output.new_zeros(
                        batch,
                        frames,
                        tile_output.size(2),
                        height * scale,
                        width * scale,
                    )
                    weight = tile_output.new_zeros(
                        batch,
                        frames,
                        1,
                        height * scale,
                        width * scale,
                    )

                crop_top = (top - padded_top) * scale
                crop_left = (left - padded_left) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_right = crop_left + (right - left) * scale
                core = tile_output[
                    :, :, :,
                    crop_top:crop_bottom,
                    crop_left:crop_right,
                ]

                output[
                    :, :, :,
                    top * scale:bottom * scale,
                    left * scale:right * scale,
                ].add_(core)
                weight[
                    :, :, :,
                    top * scale:bottom * scale,
                    left * scale:right * scale,
                ].add_(1)

        if output is None or weight is None or torch.any(weight == 0):
            raise RuntimeError('Spatial tiling left uncovered output pixels.')
        return output.div_(weight)

    @torch.no_grad()
    def _forward_device_chunk(self, net, lq_chunk, tile_opt):
        if tile_opt['use_tile']:
            return self.tile_forward_video(
                net=net,
                lq=lq_chunk,
                tile_size=tile_opt['tile_size'],
                tile_pad=tile_opt['tile_pad'],
                scale=tile_opt['scale'],
                minimum_size=tile_opt['minimum_size'],
                size_multiple=tile_opt['size_multiple'],
            )
        return self._unwrap_output(net(lq_chunk))

    @torch.no_grad()
    def _forward_sequence(self, net, lq_cpu, tile_opt):
        batch, total_frames, _, height, width = lq_cpu.shape
        if batch != 1:
            raise ValueError('Long-video inference requires batch size 1.')

        if not tile_opt['use_temporal_chunk']:
            lq_device = lq_cpu.to(self.device, non_blocking=True)
            output = self._forward_device_chunk(net, lq_device, tile_opt)
            return output.detach().cpu()

        max_frames = tile_opt['max_frames']
        temporal_pad = tile_opt['temporal_pad']
        if max_frames <= 0 or temporal_pad < 0:
            raise ValueError(
                'max_frames must be positive and temporal_pad non-negative.'
            )

        output_cpu = None
        for start in range(0, total_frames, max_frames):
            end = min(start + max_frames, total_frames)
            padded_start = max(start - temporal_pad, 0)
            padded_end = min(end + temporal_pad, total_frames)
            lq_chunk = lq_cpu[:, padded_start:padded_end].to(
                self.device,
                non_blocking=True,
            )
            chunk_output = self._forward_device_chunk(
                net,
                lq_chunk,
                tile_opt,
            )
            valid_start = start - padded_start
            valid_end = valid_start + end - start
            valid_output = chunk_output[:, valid_start:valid_end]

            if output_cpu is None:
                output_cpu = torch.empty(
                    batch,
                    total_frames,
                    valid_output.size(2),
                    height * tile_opt['scale'],
                    width * tile_opt['scale'],
                    dtype=valid_output.dtype,
                    device='cpu',
                )
            output_cpu[:, start:end].copy_(valid_output.detach().cpu())
            del lq_chunk, chunk_output, valid_output
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        return output_cpu

    def _tile_options(self):
        val_opt = self.opt.get('val') or {}
        tile = val_opt.get('tile') or {}
        return {
            'use_temporal_chunk': bool(
                tile.get('use_temporal_chunk', False)
            ),
            'max_frames': int(tile.get('max_frames', 60)),
            'temporal_pad': int(tile.get('temporal_pad', 6)),
            'use_tile': bool(tile.get('use_tile', False)),
            'tile_size': int(tile.get('tile_size', 256)),
            'tile_pad': int(tile.get('tile_pad', 32)),
            'minimum_size': int(tile.get('minimum_size', 64)),
            'size_multiple': int(tile.get('size_multiple', 4)),
            'scale': int(self.opt.get('scale', 1)),
        }

    @torch.no_grad()
    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        lq_cpu = self.lq.detach().cpu()
        tile_opt = self._tile_options()
        flip_sequence = bool((self.opt.get('val') or {}).get('flip_seq', False))

        output = self._forward_sequence(net, lq_cpu, tile_opt)
        if flip_sequence:
            reverse_output = self._forward_sequence(
                net,
                lq_cpu.flip(1),
                tile_opt,
            ).flip(1)
            output = 0.5 * (output + reverse_output)
        self.output = output

        if self.center_frame_only:
            self.output = self.output[:, self.output.size(1) // 2]
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()
