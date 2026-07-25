"""Synthesize time-varying unknown degradations for paired video training."""

import argparse
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}
DEGRADATIONS = (
    'gaussian_noise',
    'poisson_noise',
    'speckle_noise',
    'jpeg',
    'video_compression',
    'gaussian_blur',
    'resize',
)


def _video_compress(image, codecs):
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            'PyAV is required for video-compression degradation. '
            'Install the `av` requirement or use --disable-video-compression.'
        ) from exc

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as handle:
            temporary_path = Path(handle.name)
        codec = random.choice(codecs)
        with av.open(str(temporary_path), mode='w') as container:
            stream = container.add_stream(codec, rate=1)
            stream.width = image.shape[1]
            stream.height = image.shape[0]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = random.randint(10_000, 100_000)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        with av.open(str(temporary_path), mode='r') as container:
            decoded = next(container.decode(video=0)).to_ndarray(
                format='bgr24'
            )
        return decoded
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _apply_degradation(image, name, codecs):
    image_float = image.astype(np.float32) / 255.0
    if name == 'gaussian_noise':
        sigma = random.uniform(10, 15) / 255.0
        image_float += np.random.normal(0, sigma, image_float.shape)
    elif name == 'poisson_noise':
        peak = 10 ** random.uniform(2, 4)
        image_float = np.random.poisson(image_float * peak) / peak
    elif name == 'speckle_noise':
        sigma = random.uniform(10, 15) / 255.0
        image_float += image_float * np.random.normal(
            0,
            sigma,
            image_float.shape,
        )
    elif name == 'jpeg':
        quality = random.choice((20, 30, 40))
        success, encoded = cv2.imencode(
            '.jpg',
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not success:
            raise RuntimeError('OpenCV JPEG encoding failed.')
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif name == 'video_compression':
        return _video_compress(image, codecs)
    elif name == 'gaussian_blur':
        kernel = random.choice((3, 5, 7))
        sigma_x = random.uniform(0.2, 3.0)
        sigma_y = random.uniform(0.2, 3.0)
        return cv2.GaussianBlur(
            image,
            (kernel, kernel),
            sigmaX=sigma_x,
            sigmaY=sigma_y,
        )
    elif name == 'resize':
        height, width = image.shape[:2]
        resize_type = random.choices(
            ('up', 'down', 'keep'),
            weights=(0.3, 0.4, 0.3),
        )[0]
        if resize_type == 'up':
            factor = random.uniform(1.0, 2.0)
        elif resize_type == 'down':
            factor = random.uniform(0.5, 1.0)
        else:
            factor = 1.0
        interpolation = random.choice(
            (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
        )
        resized = cv2.resize(
            image,
            None,
            fx=factor,
            fy=factor,
            interpolation=interpolation,
        )
        return cv2.resize(
            resized,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        raise ValueError(f'Unknown degradation: {name}')
    return np.clip(image_float * 255.0, 0, 255).round().astype(np.uint8)


def _sequence_folders(input_root):
    folders = sorted(path for path in input_root.iterdir() if path.is_dir())
    return folders or [input_root]


def synthesize(
    input_root,
    output_root,
    continuous_frames=6,
    skip_probability=0.55,
    seed=0,
    enable_video_compression=True,
    codecs=('libx264', 'mpeg4'),
):
    """Generate LQ sequences while preserving folder and frame names."""
    if continuous_frames < 1:
        raise ValueError('continuous_frames must be positive.')
    if not 0 <= skip_probability <= 1:
        raise ValueError('skip_probability must be in [0, 1].')
    random.seed(seed)
    np.random.seed(seed)
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = list(DEGRADATIONS)
    if not enable_video_compression:
        candidates.remove('video_compression')

    total_frames = 0
    for input_folder in _sequence_folders(input_root):
        relative = (
            Path()
            if input_folder == input_root
            else input_folder.relative_to(input_root)
        )
        output_folder = output_root / relative
        output_folder.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f'Processing {input_folder} ({len(frame_paths)} frames)')
        active = []
        for frame_index, frame_path in enumerate(frame_paths):
            if frame_index % continuous_frames == 0:
                shuffled = list(candidates)
                random.shuffle(shuffled)
                active = [
                    name
                    for name in shuffled
                    if random.random() > skip_probability
                ]
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Cannot decode {frame_path}.')
            for degradation in active:
                image = _apply_degradation(image, degradation, codecs)
            destination = output_folder / frame_path.name
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f'Cannot write {destination}.')
            total_frames += 1
    print(f'Wrote {total_frames} degraded frames to {output_root}.')


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize time-varying unknown video degradations.'
    )
    parser.add_argument('--input-dir', '--input_dir', required=True)
    parser.add_argument('--output-dir', '--output_dir', required=True)
    parser.add_argument('--continuous-frames', '--continuous_frames', type=int, default=6)
    parser.add_argument('--skip-probability', '--prob', type=float, default=0.55)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--disable-video-compression',
        action='store_true',
    )
    parser.add_argument(
        '--video-codecs',
        default='libx264,mpeg4',
        help='Comma-separated PyAV encoder names.',
    )
    args = parser.parse_args()
    codecs = tuple(
        value.strip()
        for value in args.video_codecs.split(',')
        if value.strip()
    )
    synthesize(
        input_root=args.input_dir,
        output_root=args.output_dir,
        continuous_frames=args.continuous_frames,
        skip_probability=args.skip_probability,
        seed=args.seed,
        enable_video_compression=not args.disable_video_compression,
        codecs=codecs,
    )


if __name__ == '__main__':
    main()
