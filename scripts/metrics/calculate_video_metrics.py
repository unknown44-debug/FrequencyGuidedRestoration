"""Calculate per-sequence and overall PSNR/SSIM for restored videos."""

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


IMAGE_EXTENSIONS = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}


def _folders(root):
    root = Path(root)
    children = sorted(path for path in root.iterdir() if path.is_dir())
    return OrderedDict((path.name, path) for path in children) or OrderedDict([
        ('.', root)
    ])


def _images(folder):
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate(restored_root, gt_root, crop_border=0, test_y_channel=False):
    restored_folders = _folders(restored_root)
    gt_folders = _folders(gt_root)
    rows = []
    all_psnr = []
    all_ssim = []

    for folder, restored_folder in restored_folders.items():
        if folder not in gt_folders:
            raise FileNotFoundError(
                f'Missing GT sequence {folder!r} under {gt_root}.'
            )
        restored_paths = _images(restored_folder)
        gt_paths = _images(gt_folders[folder])
        if len(restored_paths) != len(gt_paths):
            raise ValueError(
                f'{folder}: {len(restored_paths)} restored frames but '
                f'{len(gt_paths)} GT frames.'
            )
        sequence_psnr = []
        sequence_ssim = []
        for restored_path, gt_path in zip(restored_paths, gt_paths):
            restored = cv2.imread(str(restored_path), cv2.IMREAD_COLOR)
            gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if restored is None or gt is None:
                raise FileNotFoundError(
                    f'Cannot decode {restored_path} or {gt_path}.'
                )
            if restored.shape != gt.shape:
                raise ValueError(
                    f'Shape mismatch: {restored_path} {restored.shape} vs '
                    f'{gt_path} {gt.shape}.'
                )
            sequence_psnr.append(calculate_psnr(
                restored,
                gt,
                crop_border=crop_border,
                input_order='HWC',
                test_y_channel=test_y_channel,
            ))
            sequence_ssim.append(calculate_ssim(
                restored,
                gt,
                crop_border=crop_border,
                input_order='HWC',
                test_y_channel=test_y_channel,
            ))
        rows.append({
            'sequence': folder,
            'frames': len(restored_paths),
            'psnr': float(np.mean(sequence_psnr)),
            'ssim': float(np.mean(sequence_ssim)),
        })
        all_psnr.extend(sequence_psnr)
        all_ssim.extend(sequence_ssim)

    overall = {
        'sequence': 'overall_frame_average',
        'frames': len(all_psnr),
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
    }
    return rows, overall


def main():
    parser = argparse.ArgumentParser(
        description='Calculate video PSNR and SSIM.'
    )
    parser.add_argument('--restored-root', '--restored_root', required=True)
    parser.add_argument('--gt-root', '--gt_root', required=True)
    parser.add_argument('--crop-border', type=int, default=0)
    parser.add_argument('--test-y-channel', action='store_true')
    parser.add_argument('--output-csv')
    args = parser.parse_args()
    rows, overall = evaluate(
        args.restored_root,
        args.gt_root,
        crop_border=args.crop_border,
        test_y_channel=args.test_y_channel,
    )
    for row in rows + [overall]:
        print(
            f'{row["sequence"]}: frames={row["frames"]}, '
            f'PSNR={row["psnr"]:.4f}, SSIM={row["ssim"]:.6f}'
        )
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=('sequence', 'frames', 'psnr', 'ssim'),
            )
            writer.writeheader()
            writer.writerows(rows + [overall])


if __name__ == '__main__':
    main()
