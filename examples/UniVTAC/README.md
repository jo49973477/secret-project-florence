# UniVTAC data for GR00T N1.7

## Overview

The UniVTAC converter produces a GR00T-flavored LeRobot v2 dataset with the existing:

- head RGB and wrist RGB
- the first eight Franka joint coordinates
- the next-timestep eight-dimensional absolute joint target
- task language

It can now optionally add one tactile RGB stream and a fixed-size point cloud. These values are transported through `LeRobotEpisodeLoader`, `extract_step_data()`, explicit `VLAStepData.tactile` / `VLAStepData.pointclouds` fields, the N1.7 processor/collator, and the optional `MultiModalConditionedDiT` action-head path.

The maintained UniVTAC alignment is unchanged: `joint[:-1, :8]` is state, `joint[1:, :8]` is action, and all observation streams use their first `T-1` samples. The stored action is still an absolute target. `ActionRepresentation.RELATIVE` in the config performs the relative transform inside GR00T.

## Expected input

The official download is arranged as task directories containing HDF5 episodes. The converter also accepts freshly collected `data/<task>/<config>/hdf5/*.hdf5` trees because it searches recursively.

The RGB baseline requires:

```text
embodiment/joint          [T, >=8]
observation/head/rgb      [T] JPEG byte streams
observation/wrist/rgb     [T] JPEG byte streams
```

Multimodal conversion additionally resolves one unmarked tactile RGB stream and one depth stream:

```text
tactile/<sensor>/rgb          [T] JPEG byte streams
tactile/<sensor>/depth        [T, 240, 320] float32
```

The semantic output is always `tactile.rgb`; the raw HDF5 path is only source provenance. If an input file contains multiple `tactile/*/rgb` datasets, select one with `--tactile-rgb-key`. When that option is omitted but `--pointcloud-depth-key` is present, the converter selects the sibling `rgb` dataset. `rgb_marker` is not silently substituted for raw `rgb`.

Install conversion dependencies with:

```bash
uv pip install h5py imageio-ffmpeg
```

The converter uses H.264 (`libx264`) in MP4. It first tries `ffmpeg`, then the self-contained `imageio-ffmpeg` executable. An explicit executable can be supplied with `--ffmpeg /path/to/ffmpeg`.

## Dataset Conversion

### RGB-only baseline

Omit the opt-in modality flags to retain the prior dataset layout and behavior:

```bash
uv run python examples/UniVTAC/convert_univtac_to_lerobot.py \
  --input /path/to/UniVTAC \
  --output /path/to/univtac_gr00t \
  --fps 10 \
  --verbose
```

### Tactile and point cloud

The inspected downloaded episode did not embed camera intrinsics or a depth-unit attribute. Therefore the source RGB/depth paths, pinhole calibration, and millimetre-to-metre scale are explicit in this example. The legacy `left_gsmini` text below is the real raw HDF5 group name; it does not create a `left` semantic modality:

```bash
uv run python examples/UniVTAC/convert_univtac_to_lerobot.py \
  --input /path/to/UniVTAC \
  --output /path/to/univtac_gr00t_multimodal \
  --fps 10 \
  --include-tactile \
  --tactile-rgb-key tactile/left_gsmini/rgb \
  --include-pointcloud \
  --pointcloud-depth-key tactile/left_gsmini/depth \
  --pointcloud-intrinsics 340.45112782 324.97607656 160 120 \
  --pointcloud-depth-scale 0.001 \
  --pointcloud-num-points 1024 \
  --verbose
```

Those calibration values come from the current UniVTAC GS Mini configuration: resolution `(320, 240)`, camera-to-surface distance `0.0283 m`, and real size `(0.0266, 0.0209) m`. Confirm them against the configuration that produced custom data. When an HDF5 depth dataset/group/file contains `intrinsics`, `camera_intrinsics`, `intrinsic_matrix`, `depth_scale`, `unit`, or equivalent supported metadata, the converter uses it and the corresponding override can be omitted. It never guesses missing calibration.

`--pointcloud-depth-key` selects one camera stream, so the example point cloud is in that selected tactile camera's OpenCV frame (`+x` right, `+y` down, `+z` along depth). It is not transformed into robot/world coordinates. Choose another depth stream and its matching intrinsics when a different frame is desired.

The output path must not exist. `--overwrite` replaces it only after a staging conversion validates. `--skip-invalid` skips malformed episodes; the default fails instead of silently dropping data.

## Output Dataset Structure

```text
univtac_gr00t_multimodal/
├── data/
│   └── chunk-000/episode_000000.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.head/episode_000000.mp4
│       ├── observation.images.wrist/episode_000000.mp4
│       └── observation.tactile.rgb/episode_000000.mp4
├── pointclouds/
│   └── chunk-000/
│       └── observation.pointcloud.xyz/episode_000000.npz
└── meta/
    ├── info.json
    ├── modality.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

Tactile RGB is physically H.264 video but remains semantically `tactile.rgb`. H.264 is lossy; decoded arrays are `uint8 [T,H,W,3]`, but pixel values need not exactly equal the source JPEG decode.

Each point-cloud NPZ contains `xyz` as `float32 [T,1024,3]`. Conversion filters non-finite and non-positive scaled depth. Valid pixels are traversed in row-major order and deterministically sampled at uniform positions. When fewer than 1024 valid pixels exist, valid points repeat deterministically; a frame with no valid point fails conversion.

## Modality Configuration

The unchanged RGB baseline remains [univtac_config.py](univtac_config.py). The experimental model-ready sensor config is [univtac_multimodal_config.py](univtac_multimodal_config.py), which adds:

```python
"tactile": ModalityConfig(
    delta_indices=[0],
    modality_keys=["rgb"],
),
"pointcloud": ModalityConfig(
    delta_indices=[0],
    modality_keys=["xyz"],
),
```

Both modalities honor their own `delta_indices` in the dataset loader. The current action-head encoders intentionally accept one observation timestep, so model training uses `[0]`; a future multi-frame encoder/aggregator is required before changing this to values such as `[-2, -1, 0]` for training.

## Generate GR00T statistics

Statistics remain state/action-oriented. External point-cloud NPZ features are excluded from parquet statistics.

```bash
uv run python gr00t/data/stats.py \
  --dataset-path /path/to/univtac_gr00t_multimodal \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_multimodal_config.py
```

For the RGB baseline, use `examples/UniVTAC/univtac_config.py` instead.

## Inspect / Verify Converted Dataset

The existing lightweight inspector validates files on disk, checks `state[0] == joint[0]` and `action[0] == joint[1]`, initializes `ShardedSingleStepDataset`, and traces one sample through `extract_step_data()` without loading model weights:

```bash
uv run python examples/UniVTAC/inspect_converted_dataset.py \
  --dataset /path/to/univtac_gr00t_multimodal \
  --episode-index 0 \
  --check-gr00t-loader \
  --modality-config examples/UniVTAC/univtac_multimodal_config.py
```

Expected sample values include:

```text
images.head           [1, 270, 480, 3] uint8
images.wrist          [1, 270, 480, 3] uint8
tactile.rgb           [1, 240, 320, 3] uint8
pointclouds.xyz       [1, 1024, 3] float32
states.joint          [1, 8] float32
actions.joint         [40, 8] float32
```

To check an existing baseline dataset in the same run, add:

```bash
  --rgb-only-dataset /path/to/existing_univtac_gr00t
```

If the original HDF5 moved, use `--source-hdf5 /new/path/to/episode.hdf5`; use `--skip-source-comparison` only when it is intentionally unavailable.

## Fine-tuning

### RGB baseline

Only the RGB baseline config is supported for current N1.7 fine-tuning:

```bash
USE_WANDB=0 NUM_GPUS=1 MAX_STEPS=100 GLOBAL_BATCH_SIZE=4 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/univtac_gr00t \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_config.py \
  --output-dir /tmp/gr00t_univtac_baseline
```

The N1.7 action horizon is 40, so a converted episode needs at least 40 aligned rows (`T >= 41`) to contribute a training sample.

<<<<<<< HEAD
### Experimental tactile + point-cloud conditioning
=======
## Baseline diagnostic evaluation

The diagnostic code is prepared and CPU-tested locally, but loading GR00T checkpoints and running the RGB ablations is intentionally left to the GPU server. After syncing or pulling this repository on the server, first run the smoke configuration on a selected physical GPU:

```bash
EVAL_GPU=5 RUN_MODE=smoke \
bash scripts/run_univtac_baseline_diagnostics.sh
```

If that succeeds, run the complete checkpoint/seed sweep:

```bash
EVAL_GPU=4 RUN_MODE=full \
bash scripts/run_univtac_baseline_diagnostics.sh
```

`EVAL_GPU` is the physical index shown by `nvidia-smi`. The script assigns it to `CUDA_VISIBLE_DEVICES`, so evaluation always uses the remapped logical device `cuda:0`; the physical index is never passed directly to torch. The defaults target `/ssdg/spl_yeongyoo/univtac_gr00t` and `outputs/univtac_rgb_joint_20260904_121012`. Override server paths and sweep settings entirely through the environment, for example:

```bash
EVAL_GPU=5 \
DATASET_PATH=/data/univtac_gr00t \
RUN_DIR=outputs/my_run \
TRAJ_IDS="0 1 2 3 4" \
SEEDS="0 1 2 3 4" \
VERIFY_REPRODUCIBILITY=1 \
RUN_MODE=full \
bash scripts/run_univtac_baseline_diagnostics.sh
```

Other overrides are `CHECKPOINT_STEPS`, `CONDITIONS`, `EVAL_STEPS`, `EXECUTION_HORIZON`, `DENOISING_STEPS`, `RESULT_DIR`, and `EMBODIMENT_TAG`. Full mode defaults to checkpoints `500 1000 1500 2000`, trajectories `0 1 2 3 4`, seeds `0 1 2 3 4`, all three conditions, and 400 steps. Smoke mode defaults to checkpoint 500, trajectory 0, seed 0, all conditions, and 32 steps. Both modes use an execution horizon of 16 and four denoising steps.

The result directory contains:

```text
diagnostics/
├── logs/
├── raw_metrics.csv
├── summary_metrics.csv
├── persistence_metrics.csv
└── reproducibility_check.csv  # only with VERIFY_REPRODUCIBILITY=1
```

`raw_metrics.csv` has one row per checkpoint, RGB condition, and seed. `summary_metrics.csv` reports mean and population standard deviation across seeds, plus absolute and percentage degradation relative to normal RGB. `persistence_metrics.csv` reports both one-step persistence (`q_t` predicts `q_(t+1)`) and chunk-hold persistence (repeat `q_t` for the execution horizon). Logs include the complete configuration and detailed progress. With `VERIFY_REPRODUCIBILITY=1`, the first checkpoint/condition/seed combination is repeated and its exact and tolerance-based metric matches are saved.

Interpret these offline diagnostics conservatively. Similar normal, black, and shuffled results suggest that the current policy is weakly dependent on RGB under this offline test; significantly worse ablations indicate that RGB provides predictive information. GR00T performance near persistence suggests that temporal or proprioceptive smoothness may explain much of the score, while a substantial improvement over persistence indicates predictive structure beyond trivial persistence. These tests alone do not establish that RGB is useless or that tactile sensing is necessary.

## Common failures
>>>>>>> origin/rgbd_test

Select the multimodal dataset config and DiT explicitly. This example freezes the pre-existing VLM, projector/action path, VLM normalization, and base DiT while training only the point/tactile encoders, sensor cross-attention branches, and residual gates:

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/univtac_gr00t_multimodal \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_multimodal_config.py \
  --dit-type multimodal_conditioned_dit \
  --point-input-dim 3 \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --no-tune-vlln \
  --no-tune-diffusion-model \
  --tune-point-encoder \
  --tune-tactile-encoder \
  --tune-multimodal-adapter \
  --output-dir /tmp/gr00t_univtac_multimodal
```

Use `--no-use-point-conditioning` or `--no-use-tactile-conditioning` for a single-sensor ablation. The UniVTAC conversion currently provides XYZ only, hence `--point-input-dim 3`; a dataset/config containing aligned XYZ and RGB point features can use 6.

## Current Limitations

The extension includes a ResNet-18 tactile encoder, lightweight PointNet++/point-transformer choices, and gated action-token cross-attention. It does **not** include PTv3, PointACT-style fusion, a production-scale point-cloud architecture, cross-modal fusion beyond the existing independent residual branches, trained multimodal weights, or multimodal open-loop evaluation. Tactile input is scaled to `[0, 1]`; point coordinates remain in the selected depth camera frame and receive no dataset-statistics normalization.

## Backward Compatibility

Existing RGB + state + action + language datasets do not need `tactile`, `pointcloud`, or `pointcloud_path` metadata. Conversion without `--include-tactile` and `--include-pointcloud` produces the prior layout. Continue using [univtac_config.py](univtac_config.py) with the default `alternate_vl_dit`; only use [univtac_multimodal_config.py](univtac_multimodal_config.py) with matching converted modalities and `--dit-type multimodal_conditioned_dit`.

Common failures remain fail-fast: missing head/wrist/tactile streams, temporal-length mismatches, odd H.264 dimensions, invalid/missing depth calibration, a frame with no valid depth, or a point-cloud NPZ whose row count differs from the episode DataFrame.
