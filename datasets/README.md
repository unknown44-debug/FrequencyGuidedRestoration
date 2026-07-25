# Dataset layout

Dataset files are intentionally excluded from Git. The provided options expect:

```text
datasets/
├── DAVIS/
│   ├── train/
│   │   ├── GT/<sequence>/<frame>.jpg
│   │   └── LQ/<sequence>/<frame>.jpg
│   ├── test/
│   │   ├── GT/<sequence>/<frame>.png
│   │   └── T6/blur/<sequence>/<frame>.png
│   └── meta_info_DAVIS_train.txt
├── Set8/
│   ├── GT/<sequence>/<frame>.png
│   └── T6/blur/<sequence>/<frame>.png
└── FROTH/
    ├── gt/<sequence>/<frame>.png
    └── blur/<sequence>/<frame>.png
```

Generate synthesized DAVIS/TUD LQ frames:

```bash
python scripts/data_preparation/synthesize_tud_dataset.py \
  --input-dir datasets/DAVIS/train/GT \
  --output-dir datasets/DAVIS/train/LQ \
  --continuous-frames 6
```

Generate the training metadata:

```bash
python scripts/data_preparation/generate_video_meta_info.py \
  --dataset-path datasets/DAVIS/train/GT \
  --output-path datasets/DAVIS/meta_info_DAVIS_train.txt
```

Metadata lines have the form:

```text
sequence_name frame_count (height,width,channels) start_frame
```

Frame stems must be numeric. Change `filename_tmpl` and `filename_ext` in the
training YAML if the dataset does not use five-digit JPEG names.
