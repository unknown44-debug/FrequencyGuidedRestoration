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
[test dataset](https://drive.google.com/drive/folders/1-3i3Gm48APnQ3tsNs9ANj9OiApBHTRgH?usp=sharing "Davis&Set-8") and 
[train dataset](https://pan.baidu.com/s/15xR24T1-ktnJQYl6PQy-Tw?pwd=xk27 "Davis2017")
[GoPro-TUD&DVD-TUD dataset]( https://pan.baidu.com/s/1a4UuT12LE2mBzMaj6MciAQ?pwd=tmwm "GoPro-TUD&DVD-TUD")

Download the pre-trained weight&states from：[pre-trained weight](https://pan.baidu.com/s/1OcSJ-Az8y_oi5Ll9ezVbNQ?pwd=ygqn "trained on Davis2017")

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

## Visualization

![Davis&set-8](https://github.com/unknown44-debug/FrequencyGuidedRestoration/blob/main/Visualization/github.png)

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



## License and acknowledgement

The repository license is Apache-2.0. This applies only to material for which
the repository owner has the right to grant that license.

This project is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and
uses [SPyNet](https://spynet.is.tue.mpg.de/) for optical-flow estimation. We
thank the authors for making their code and models available.

The design of this work was inspired by
[AverNet](https://github.com/XLearning-SCU/2024-NeurIPS-AverNet),
[AdaIR](https://github.com/c-yn/AdaIR), and the grouped spatial-temporal shift
design in [Shift-Net (GShift)](https://github.com/dasongli1/Shift-Net).

Please cite the related works when using this repository:

```bibtex
@misc{basicsr,
  author       = {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C. K. Chan
                  and Chen Change Loy and Chao Dong},
  title        = {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year         = {2022}
}

@inproceedings{ranjan2017spynet,
  author    = {Anurag Ranjan and Michael J. Black},
  title     = {Optical Flow Estimation Using a Spatial Pyramid Network},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2017},
  pages     = {4161--4170}
}

@inproceedings{zhao2024avernet,
  author    = {Haiyu Zhao and Lei Tian and Xinyan Xiao and Peng Hu and Yuanbiao Gou and Xi Peng},
  title     = {{AverNet}: All-in-one Video Restoration for Time-varying Unknown Degradations},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024}
}

@inproceedings{cui2025adair,
  author    = {Yuning Cui and Syed Waqas Zamir and Salman Khan and Alois Knoll
               and Mubarak Shah and Fahad Shahbaz Khan},
  title     = {{AdaIR}: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}

@inproceedings{li2023shift,
  author    = {Dasong Li and Xiaoyu Shi and Yi Zhang and Ka Chun Cheung and
               Simon See and Xiaogang Wang and Hongwei Qin and Hongsheng Li},
  title     = {A Simple Baseline for Video Restoration With Grouped Spatial-Temporal Shift},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023},
  pages     = {9822--9832}
}
```

Add the FrequencyGuidedRestoration paper citation here before release.
