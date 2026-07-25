"""Training and standard inference wrapper for frequency-guided restoration."""

from collections import Counter
from os import path as osp

import torch
from torch import distributed as dist
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class FrequencyGuidedVideoRestorationModel(VideoBaseModel):
    """BasicSR model wrapper for recurrent all-frame video restoration.

    The wrapper preserves BasicSR's optimizer, scheduler, EMA, checkpoint, and
    logging behavior while adding motion-estimator learning-rate control and
    sequence-aware validation.
    """

    _MOTION_NAMES = ('motion_estimator', 'spynet')
    _WARMUP_NAMES = ('motion_estimator', 'spynet', 'feat_extract', 'edvr')

    def __init__(self, opt):
        super().__init__(opt)
        val_opt = opt.get('val') or {}
        self.center_frame_only = bool(val_opt.get('center_frame_only', False))
        self.fix_motion_iter = (
            opt.get('train', {}).get('fix_motion')
            if self.is_train
            else None
        )
        if self.fix_motion_iter is None and self.is_train:
            # Backward-compatible option used by the experimental framework.
            self.fix_motion_iter = opt.get('train', {}).get('fix_flow')

    @classmethod
    def _is_motion_parameter(cls, name):
        return any(token in name for token in cls._MOTION_NAMES)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        base_lr = train_opt['optim_g']['lr']
        motion_lr_mul = train_opt.get(
            'motion_lr_mul',
            train_opt.get('flow_lr_mul', 1),
        )
        logger = get_root_logger()
        logger.info(
            'Use motion-estimator learning-rate multiplier %.4f.',
            motion_lr_mul,
        )

        normal_params = []
        motion_params = []
        for name, parameter in self.net_g.named_parameters():
            if self._is_motion_parameter(name):
                motion_params.append(parameter)
            else:
                normal_params.append(parameter)

        if motion_lr_mul == 1 or not motion_params:
            optim_params = normal_params + motion_params
        else:
            optim_params = []
            if normal_params:
                optim_params.append({'params': normal_params, 'lr': base_lr})
            optim_params.append({
                'params': motion_params,
                'lr': base_lr * motion_lr_mul,
            })

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type,
            optim_params,
            **train_opt['optim_g'],
        )
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_motion_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(
                    'Freeze motion estimator and feature extractor for %d iterations.',
                    self.fix_motion_iter,
                )
                for name, parameter in self.net_g.named_parameters():
                    if any(token in name for token in self._WARMUP_NAMES):
                        parameter.requires_grad_(False)
            elif current_iter == self.fix_motion_iter:
                logger.warning('Unfreeze all generator parameters.')
                self.net_g.requires_grad_(True)

        super().optimize_parameters(current_iter)

    @staticmethod
    def _unwrap_output(output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(
                f'Generator must return a tensor or tensor sequence, got {type(output)!r}.'
            )
        return output

    @torch.no_grad()
    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        original_frames = self.lq.size(1) if self.lq.dim() == 5 else None
        flip_sequence = bool((self.opt.get('val') or {}).get('flip_seq', False))

        input_lq = self.lq
        if flip_sequence and original_frames is not None:
            input_lq = torch.cat([input_lq, input_lq.flip(1)], dim=1)

        self.output = self._unwrap_output(net(input_lq))

        if flip_sequence and original_frames is not None:
            output_forward = self.output[:, :original_frames]
            output_reverse = self.output[:, original_frames:].flip(1)
            self.output = 0.5 * (output_forward + output_reverse)

        if self.center_frame_only and self.output.dim() == 5:
            self.output = self.output[:, original_frames // 2]

        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

    @staticmethod
    def _frame_counts(dataset):
        if hasattr(dataset, 'frame_counts'):
            return dict(dataset.frame_counts)
        if hasattr(dataset, 'data_info') and 'folder' in dataset.data_info:
            return dict(Counter(dataset.data_info['folder']))
        raise AttributeError(
            'Video validation dataset must expose frame_counts or data_info["folder"].'
        )

    def _prepare_metric_storage(self, dataset, dataset_name, metric_opts):
        frame_counts = self._frame_counts(dataset)
        if self.center_frame_only:
            frame_counts = {
                folder: 1 for folder in frame_counts
            }
        metric_count = len(metric_opts)
        rebuild = not hasattr(self, 'metric_results')
        if not rebuild:
            rebuild = set(self.metric_results) != set(frame_counts)
        if not rebuild:
            rebuild = any(
                self.metric_results[folder].shape != (count, metric_count)
                for folder, count in frame_counts.items()
            )
        if rebuild:
            self.metric_results = {
                folder: torch.zeros(
                    count,
                    metric_count,
                    dtype=torch.float32,
                    device=self.device,
                )
                for folder, count in frame_counts.items()
            }
        for tensor in self.metric_results.values():
            tensor.zero_()
        self._initialize_best_metric_results(dataset_name)

    def dist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img,
    ):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        val_opt = self.opt.get('val') or {}
        metric_opts = val_opt.get('metrics')
        with_metrics = bool(metric_opts)
        rank, world_size = get_dist_info()

        if with_metrics:
            self._prepare_metric_storage(dataset, dataset_name, metric_opts)

        num_folders = len(dataset)
        num_pad = (world_size - num_folders % world_size) % world_size
        pbar = tqdm(total=num_folders, unit='folder') if rank == 0 else None

        for i in range(rank, num_folders + num_pad, world_size):
            dataset_idx = min(i, num_folders - 1)
            val_data = dataset[dataset_idx]
            folder = val_data['folder']
            batched_data = dict(val_data)
            batched_data['lq'] = val_data['lq'].unsqueeze(0)
            if 'gt' in val_data:
                batched_data['gt'] = val_data['gt'].unsqueeze(0)

            self.feed_data(batched_data)
            self.test()
            visuals = self.get_current_visuals()

            result_sequence = visuals['result']
            gt_sequence = visuals.get('gt')
            if self.center_frame_only:
                result_sequence = result_sequence.unsqueeze(1)
                if gt_sequence is not None and gt_sequence.dim() == 5:
                    gt_sequence = gt_sequence[
                        :, gt_sequence.size(1) // 2
                    ].unsqueeze(1)

            if i < num_folders:
                for frame_idx in range(result_sequence.size(1)):
                    result_img = tensor2img([
                        result_sequence[0, frame_idx]
                    ])
                    metric_data = {'img': result_img}
                    if gt_sequence is not None:
                        metric_data['img2'] = tensor2img([
                            gt_sequence[0, frame_idx]
                        ])

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError(
                                'Saving validation frames during training is unsupported.'
                            )
                        suffix = val_opt.get('suffix') or self.opt['name']
                        img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            folder,
                            f'{frame_idx:08d}_{suffix}.png',
                        )
                        imwrite(result_img, img_path)

                    if with_metrics:
                        if gt_sequence is None:
                            raise RuntimeError(
                                'Full-reference metrics require ground-truth frames.'
                            )
                        for metric_idx, metric_opt in enumerate(
                            metric_opts.values()
                        ):
                            value = calculate_metric(metric_data, metric_opt)
                            self.metric_results[folder][
                                frame_idx, metric_idx
                            ] += value

                if pbar is not None:
                    for _ in range(world_size):
                        if pbar.n < num_folders:
                            pbar.update(1)
                    pbar.set_description(f'Folder: {folder}')

            for attr in ('lq', 'gt', 'output'):
                if hasattr(self, attr):
                    delattr(self, attr)
            del visuals
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if pbar is not None:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                for tensor in self.metric_results.values():
                    dist.reduce(tensor, 0)
                dist.barrier()
            if rank == 0:
                self._log_validation_metric_values(
                    current_iter,
                    dataset_name,
                    tb_logger,
                )

    def nondist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img,
    ):
        self.dist_validation(
            dataloader,
            current_iter,
            tb_logger,
            save_img,
        )
