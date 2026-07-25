"""Paired video datasets for training and full-sequence evaluation."""

import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils import data as data

from basicsr.data.data_util import read_img_seq
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def _parse_meta_info(path):
    """Parse ``folder count (h,w,c) [start]`` metadata."""
    clips = []
    with open(path, 'r', encoding='utf-8') as meta_file:
        for line_number, raw_line in enumerate(meta_file, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) not in (3, 4):
                raise ValueError(
                    f'{path}:{line_number}: expected 3 or 4 fields, got {line!r}.'
                )
            folder = fields[0]
            frame_count = int(fields[1])
            start_frame = int(fields[3]) if len(fields) == 4 else 0
            if frame_count <= 0:
                raise ValueError(
                    f'{path}:{line_number}: frame count must be positive.'
                )
            clips.append((folder, frame_count, start_frame))
    if not clips:
        raise ValueError(f'No clips found in metadata file: {path}')
    return clips


@DATASET_REGISTRY.register()
class PairedVideoRestorationDataset(data.Dataset):
    """Paired recurrent training dataset with temporal window sampling.

    Expected layout:

    .. code-block:: text

        dataroot_gt/clip_name/00000.jpg
        dataroot_lq/clip_name/00000.jpg
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = int(opt.get('scale', 1))
        self.gt_size = int(opt['gt_size'])
        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq'])
        self.num_frame = int(opt['num_frame'])
        self.filename_tmpl = str(opt.get('filename_tmpl', '05d'))
        self.filename_ext = str(opt.get('filename_ext', 'jpg')).lstrip('.')
        self.interval_list = [
            int(value) for value in opt.get('interval_list', [1])
        ]
        self.random_reverse = bool(opt.get('random_reverse', False))
        self.use_hflip = bool(opt.get('use_hflip', True))
        self.use_rot = bool(opt.get('use_rot', True))
        if self.num_frame < 1:
            raise ValueError('num_frame must be positive.')
        if not self.interval_list or min(self.interval_list) < 1:
            raise ValueError('interval_list must contain positive integers.')

        self.clips = _parse_meta_info(opt['meta_info_file'])
        self.samples = []
        for folder, frame_count, start_frame in self.clips:
            for anchor in range(start_frame, start_frame + frame_count):
                self.samples.append(
                    (folder, frame_count, start_frame, anchor)
                )

        self.file_client = None
        self.io_backend_opt = dict(opt.get('io_backend') or {'type': 'disk'})
        self.is_lmdb = self.io_backend_opt.get('type') == 'lmdb'
        if self.is_lmdb:
            self.io_backend_opt['db_paths'] = [
                str(self.lq_root),
                str(self.gt_root),
            ]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

    def _read_pair(self, folder, frame_index):
        stem = f'{frame_index:{self.filename_tmpl}}'
        relative_path = f'{folder}/{stem}'
        if self.is_lmdb:
            lq_path = relative_path
            gt_path = relative_path
        else:
            filename = f'{stem}.{self.filename_ext}'
            lq_path = str(self.lq_root / folder / filename)
            gt_path = str(self.gt_root / folder / filename)
        img_lq = imfrombytes(
            self.file_client.get(lq_path, 'lq'),
            float32=True,
        )
        img_gt = imfrombytes(
            self.file_client.get(gt_path, 'gt'),
            float32=True,
        )
        return img_lq, img_gt, gt_path

    def __getitem__(self, index):
        if self.file_client is None:
            backend = self.io_backend_opt.pop('type')
            self.file_client = FileClient(
                backend,
                **self.io_backend_opt,
            )

        folder, frame_count, first_frame, anchor = self.samples[index]
        interval = random.choice(self.interval_list)
        required_span = (self.num_frame - 1) * interval + 1
        if frame_count < required_span:
            raise ValueError(
                f'Clip {folder!r} has {frame_count} frames, but '
                f'{required_span} are required for num_frame={self.num_frame} '
                f'and interval={interval}.'
            )

        last_start = first_frame + frame_count - required_span
        start = min(max(anchor, first_frame), last_start)
        neighbor_list = [
            start + offset * interval for offset in range(self.num_frame)
        ]
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        img_lqs = []
        img_gts = []
        gt_path = None
        for frame_index in neighbor_list:
            img_lq, img_gt, gt_path = self._read_pair(
                folder,
                frame_index,
            )
            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        img_gts, img_lqs = paired_random_crop(
            img_gts,
            img_lqs,
            self.gt_size,
            self.scale,
            gt_path,
        )
        combined = augment(
            img_lqs + img_gts,
            hflip=self.use_hflip,
            rotation=self.use_rot,
        )
        tensors = img2tensor(combined)
        split = len(img_lqs)
        img_lqs = torch.stack(tensors[:split], dim=0)
        img_gts = torch.stack(tensors[split:], dim=0)
        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': f'{folder}/{neighbor_list[0]:{self.filename_tmpl}}',
        }

    def __len__(self):
        return len(self.samples)


def _image_files(folder):
    extensions = {
        '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
    }
    return sorted(
        str(path)
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _sequence_folders(root):
    root = Path(root)
    children = sorted(path for path in root.iterdir() if path.is_dir())
    if children:
        return OrderedDict((path.name, path) for path in children)
    if _image_files(root):
        return OrderedDict([('.', root)])
    raise ValueError(f'No video frames found under {root}.')


@DATASET_REGISTRY.register()
class PairedVideoRecurrentTestDataset(data.Dataset):
    """Return one complete LQ/GT sequence per item.

    Unlike BasicSR 1.4.2's recurrent test dataset, both cached and on-demand
    loading are supported. Set ``cache_data: false`` for long videos.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cache_data = bool(opt.get('cache_data', False))
        self.lq_root = Path(opt['dataroot_lq'])
        gt_value = opt.get('dataroot_gt')
        self.gt_root = Path(gt_value) if gt_value else None

        lq_folders = _sequence_folders(self.lq_root)
        gt_folders = (
            _sequence_folders(self.gt_root)
            if self.gt_root is not None
            else None
        )
        requested_folders = None
        if opt.get('meta_info_file'):
            requested_folders = {
                folder
                for folder, _, _ in _parse_meta_info(
                    opt['meta_info_file']
                )
            }

        self.sequences = []
        self.frame_counts = OrderedDict()
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'folder': [],
            'idx': [],
            'border': [],
        }
        for folder, lq_folder in lq_folders.items():
            if requested_folders is not None and folder not in requested_folders:
                continue
            if gt_folders is not None and folder not in gt_folders:
                raise FileNotFoundError(
                    f'Missing GT sequence {folder!r} under {self.gt_root}.'
                )
            lq_paths = _image_files(lq_folder)
            gt_paths = (
                _image_files(gt_folders[folder])
                if gt_folders is not None
                else None
            )
            if gt_paths is not None and len(lq_paths) != len(gt_paths):
                raise ValueError(
                    f'Sequence {folder!r}: {len(lq_paths)} LQ frames but '
                    f'{len(gt_paths)} GT frames.'
                )
            frame_count = len(lq_paths)
            self.frame_counts[folder] = frame_count
            self.data_info['lq_path'].extend(lq_paths)
            self.data_info['gt_path'].extend(gt_paths or [''] * frame_count)
            self.data_info['folder'].extend([folder] * frame_count)
            self.data_info['idx'].extend(
                f'{idx}/{frame_count}' for idx in range(frame_count)
            )
            self.data_info['border'].extend([0] * frame_count)
            if self.cache_data:
                lq_value = read_img_seq(lq_paths)
                gt_value = read_img_seq(gt_paths) if gt_paths else None
            else:
                lq_value = lq_paths
                gt_value = gt_paths
            self.sequences.append(
                (folder, lq_value, gt_value, lq_paths)
            )
        if not self.sequences:
            raise ValueError('No matching video sequences were found.')

    def __getitem__(self, index):
        folder, lq_value, gt_value, lq_paths = self.sequences[index]
        if self.cache_data:
            imgs_lq = lq_value
            imgs_gt = gt_value
        else:
            imgs_lq = read_img_seq(lq_value)
            imgs_gt = read_img_seq(gt_value) if gt_value else None
        result = {
            'lq': imgs_lq,
            'folder': folder,
            'lq_path': lq_paths,
        }
        if imgs_gt is not None:
            result['gt'] = imgs_gt
        return result

    def __len__(self):
        return len(self.sequences)
