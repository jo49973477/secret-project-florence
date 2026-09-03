# UniVTAC baseline for GR00T N1.7

This example converts raw [UniVTAC](https://github.com/univtac/UniVTAC) HDF5 episodes into the GR00T-flavored LeRobot v2 format. The baseline deliberately contains only:

- head RGB
- wrist RGB
- the first eight Franka joint coordinates
- the next-timestep eight-dimensional absolute joint target
- one task-language annotation

Tactile images, depth, point clouds, and end-effector values are not converted.

## Expected input

The official dataset download uses this layout:

```text
UniVTAC/
├── lift_bottle/
│   └── clean/
│       ├── 0.hdf5
│       └── 1.hdf5
├── insert_HDMI/
│   └── clean/
│       └── 0.hdf5
└── ...
```

Freshly collected data may instead contain `data/<task>/<config>/hdf5/*.hdf5`. Both layouts are accepted because the converter searches a directory recursively.

Each raw episode must contain:

```text
embodiment/joint          [T, >=8]
observation/head/rgb      [T] JPEG byte streams
observation/wrist/rgb     [T] JPEG byte streams
```

UniVTAC's maintained preprocessing uses `joint[:-1, :8]` as state, `joint[1:, :8]` as action, and the first `T-1` camera frames. The converted `action` remains an absolute joint target. `ActionRepresentation.RELATIVE` in [univtac_config.py](univtac_config.py) performs the relative transform inside GR00T.

## Install conversion dependencies

`h5py` is needed only for UniVTAC conversion and is not a core GR00T dependency:

```bash
uv pip install h5py imageio-ffmpeg
```

The converter uses H.264 (`libx264`) in an MP4 container. It first looks for `ffmpeg` and can fall back to the executable bundled by `imageio-ffmpeg`. An explicit executable can be supplied with `--ffmpeg /path/to/ffmpeg`.

## Convert

Convert a directory tree:

```bash
uv run python examples/UniVTAC/convert_univtac_to_lerobot.py \
  --input /path/to/UniVTAC \
  --output /path/to/univtac_gr00t \
  --fps 10 \
  --verbose
```

Convert one episode for debugging:

```bash
uv run python examples/UniVTAC/convert_univtac_to_lerobot.py \
  --input /path/to/lift_bottle/clean/0.hdf5 \
  --output /tmp/univtac_gr00t_test \
  --task "lift the bottle" \
  --max-episodes 1
```

The output path must not exist. Pass `--overwrite` to replace an existing dataset after the newly converted staging dataset passes validation. Conversion fails on malformed episodes by default; `--skip-invalid` is available for a large tree containing known corrupt files.

## Inspect

The converter records each original HDF5 path in `meta/episodes.jsonl`, allowing the inspector to verify the first converted state/action pair against its source:

```bash
uv run python examples/UniVTAC/inspect_converted_dataset.py \
  --dataset /path/to/univtac_gr00t \
  --episode-index 0
```

If the source dataset moved, pass `--source-hdf5 /new/path/to/0.hdf5`. Use `--skip-source-comparison` only when the source is intentionally unavailable.

## Generate GR00T statistics

The current statistics entry point uses Tyro and accepts these exact flags:

```bash
uv run python gr00t/data/stats.py \
  --dataset-path /path/to/univtac_gr00t \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_config.py
```

This writes `meta/stats.json` and `meta/relative_stats.json`. Because the dataset stores absolute `q_(t+1)` targets, relative statistics are computed by GR00T using the current `q_t` state.

After generating statistics, exercise dataset initialization, both videos, language resolution, and one action chunk through the actual GR00T loader:

```bash
uv run python examples/UniVTAC/inspect_converted_dataset.py \
  --dataset /path/to/univtac_gr00t \
  --episode-index 0 \
  --check-gr00t-loader
```

## Minimal fine-tuning

The N1.7 base checkpoint has a 40-step action horizon. Accordingly, `univtac_config.py` uses `range(40)`, even though UniVTAC's ACT baseline uses chunks of 50; the current N1.7 processor rejects a 50-step modality horizon before training.

```bash
USE_WANDB=0 NUM_GPUS=1 MAX_STEPS=100 GLOBAL_BATCH_SIZE=4 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/univtac_gr00t \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_config.py \
  --output-dir /tmp/gr00t_univtac_baseline
```

Each converted episode needs at least 40 aligned rows (`T >= 41`) to contribute a training sample with this configuration.

## Common failures

- **Missing `h5py`:** run `uv pip install h5py` in the environment used for conversion and inspection.
- **FFmpeg is missing or lacks `libx264`:** install a full FFmpeg build, install `imageio-ffmpeg`, or pass `--ffmpeg` explicitly.
- **Missing head or wrist stream:** this baseline requires both cameras and fails instead of silently substituting a view.
- **Length mismatch:** `embodiment/joint`, head RGB, and wrist RGB must all contain exactly `T` samples. The converter will not truncate mismatched data silently.
- **Odd image dimensions:** H.264 `yuv420p` requires even width and height. Resize the source deliberately rather than allowing an implicit geometry change.
- **No usable training samples:** with a 40-step action horizon, an episode must have at least 40 converted frames.
- **Loader says statistics are missing:** run `gr00t/data/stats.py` before `--check-gr00t-loader` or fine-tuning.
- **Source comparison fails after moving data:** pass the matching raw episode through `--source-hdf5`.
- **Gated backbone access fails during fine-tuning:** request access to `nvidia/Cosmos-Reason2-2B` and authenticate with `hf auth login`.
