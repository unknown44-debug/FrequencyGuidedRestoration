# FrequencyGuidedRestoration

Frequency-guided multi-range video restoration built on
[BasicSR](https://github.com/XPixelGroup/BasicSR). The repository contains the
final network in one architecture file, paired recurrent video datasets,
memory-bounded long-video inference, DAVIS/TUD training options, DAVIS/Set8/FROTH
test options, data synthesis, and PSNR/SSIM evaluation.

## Method

The generator combines:

1. global and local high-frequency prompt interaction;
2. frequency-guided second-order propagation;
3. bidirectional multi-range temporal routing with sparse historical states;
4. progressive gated reconstruction.

The registered generator is `FrequencyGuidedMultiRangeRestorationNet`.
`FrequencyGuidedVideoRestorationModel` provides standard training/inference, and
`FrequencyGuidedLongVideoModel` adds temporal chunks and spatial tiles while
keeping complete long sequences in CPU memory.

## Installation

Python 3.8 or newer is required. Install a PyTorch/torchvision pair matching the
CUDA toolkit first, then install this repository:

```bash
git clone https://github.com/unknown44-debug/FrequencyGuidedRestoration.git
cd FrequencyGuidedRestoration
python -m pip install -r requirements.txt
python -m pip install -e .
```

The deformable convolution uses `torchvision.ops.deform_conv2d`; MMCV is not
required.

Verify the included SPyNet checkpoint:

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('experiments/pretrained_models/spynet_sintel_final.pth'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected digest:
`3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714`.

## Data preparation

See [`datasets/README.md`](datasets/README.md) for the expected directory
layout. To synthesize time-varying unknown degradations and generate metadata:

```bash
python scripts/data_preparation/synthesize_tud_dataset.py \
  --input-dir datasets/DAVIS/train/GT \
  --output-dir datasets/DAVIS/train/LQ \
  --continuous-frames 6

python scripts/data_preparation/generate_video_meta_info.py \
  --dataset-path datasets/DAVIS/train/GT \
  --output-path datasets/DAVIS/meta_info_DAVIS_train.txt
```
Or Download the dataset from：

test dataset
https://drive.google.com/drive/folders/1-3i3Gm48APnQ3tsNs9ANj9OiApBHTRgH?usp=sharing

train dataset
https://pan.baidu.com/s/15xR24T1-ktnJQYl6PQy-Tw?pwd=xk27

## Training

Edit dataset paths if necessary, then run:

```bash
python basicsr/train.py \
  -opt options/train/train_frequency_guided_davis_tud.yml
```

Distributed training:

```bash
torchrun --nproc_per_node=NUM_GPUS basicsr/train.py \
  -opt options/train/train_frequency_guided_davis_tud.yml \
  --launcher pytorch
```

The training configuration retains the main settings from
`train_FSPMResidualHardSparseHistoryBSSTNet.yml`: 15-frame samples, 96 feature
channels, 5 extraction blocks, 7 propagation blocks, two sparse-history routing
layers, Adam, cosine restart scheduling, EMA, and a Charbonnier loss. Obsolete
experimental option names were translated to the public class API.

## Evaluation

Place the generator checkpoint at
`experiments/pretrained_models/frequency_guided_restoration.pth` or edit
`pretrain_network_g`. Run one of:

```bash
python basicsr/test.py -opt options/test/test_frequency_guided_davis_t6.yml
python basicsr/test.py -opt options/test/test_frequency_guided_set8_t6.yml
python basicsr/test.py -opt options/test/test_frequency_guided_froth.yml
```

The test options use `cache_data: false`, 60-frame temporal chunks with six
context frames, and 256×256 spatial tiles with 32-pixel context. These values can
be changed under `val.tile`.

Calculate metrics for an existing result directory:

```bash
python scripts/metrics/calculate_video_metrics.py \
  --restored-root results/frequency_guided_set8_t6/visualization/Set8_T6 \
  --gt-root datasets/Set8/GT \
  --output-csv results/set8_metrics.csv
```

## Checkpoint migration

The generator translates the module names used by the earlier
`FSPMResidualHardSparseHistoryBSSTNet` checkpoint (`spynet`,
prompt/alignment, sparse-history, router) when `load_state_dict` is called. The
public architecture keeps the trained `32/128`-channel reconstruction layout:
under equivalent options, all 474 state tensors map one-to-one and the migrated
network produces the same smoke-test output. `strict_load_g: true` remains
enabled so a genuinely incompatible checkpoint is reported rather than partly
loaded.

## Repository layout

```text
FrequencyGuidedRestoration/
├── basicsr/
│   ├── archs/frequency_guided_multi_range_restoration_arch.py
│   ├── models/
│   │   ├── frequency_guided_video_restoration_model.py
│   │   └── frequency_guided_long_video_model.py
│   └── data/paired_video_restoration_dataset.py
├── options/{train,test}/
├── scripts/{data_preparation,metrics}/
├── datasets/README.md
├── experiments/pretrained_models/
├── requirements.txt
├── LICENSE
└── NOTICE
```

BasicSR framework files used by training and evaluation remain based on the
official BasicSR 1.4.2 distribution.

## Before publishing

See [`NOTICE`](NOTICE) for attribution and pretrained-weight notes.

## License and acknowledgement

The repository license is Apache-2.0. This applies only to material for which
the repository owner has the right to grant that license.

The framework is based on BasicSR. Please cite BasicSR and the original SPyNet
work when using their implementation/checkpoint. Add the FrequencyGuidedRestoration
paper citation here before release.
