"""Generate BasicSR-compatible metadata for a folder of video sequences."""

import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}


def generate_meta_info(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    sequence_folders = sorted(
        path for path in dataset_path.iterdir() if path.is_dir()
    )
    if not sequence_folders:
        sequence_folders = [dataset_path]

    lines = []
    total_frames = 0
    for folder in sequence_folders:
        frames = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not frames:
            continue
        with Image.open(frames[0]) as image:
            width, height = image.size
            channels = len(image.getbands())
        try:
            start_frame = int(frames[0].stem)
        except ValueError as exc:
            raise ValueError(
                f'First frame name must be numeric, got {frames[0].name!r}.'
            ) from exc
        folder_name = '.' if folder == dataset_path else folder.name
        lines.append(
            f'{folder_name} {len(frames)} '
            f'({height},{width},{channels}) {start_frame}'
        )
        total_frames += len(frames)
        print(
            f'{folder_name}: {len(frames)} frames, '
            f'start={start_frame}, shape=({height},{width},{channels})'
        )

    if not lines:
        raise ValueError(f'No video frames found under {dataset_path}.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(
        f'Wrote {len(lines)} sequences / {total_frames} frames '
        f'to {output_path}.'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate paired-video metadata.'
    )
    parser.add_argument('--dataset-path', '--dataset_path', required=True)
    parser.add_argument('--output-path', '--output_path', required=True)
    args = parser.parse_args()
    generate_meta_info(args.dataset_path, args.output_path)


if __name__ == '__main__':
    main()
