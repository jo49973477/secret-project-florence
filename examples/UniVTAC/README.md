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
    delta_indices=[-1, 0],
    modality_keys=["rgb"],
),
"pointcloud": ModalityConfig(
    delta_indices=[0],
    modality_keys=["scene"],
),
```

Both modalities honor their own `delta_indices` in the dataset loader. UniVTAC is nominally 10 Hz, so tactile `[-1,0]` is a 100 ms pair. The official Sparsh checkpoint used `I_t ⊕ I_(t-5)` at 60 Hz (about 83 ms); the encoder receives chronological `[previous,current]` frames and reorders them internally to the official current-then-previous six-channel input. For a dataset with FPS `f`, choose a negative delta near `round(0.083 * f)` frames and keep the current frame at zero.

At an episode boundary, padded sampling repeats frame zero; unpadded training excludes reference steps whose deltas leave the episode. Negative pandas indices are rejected, so an episode can never borrow the previous episode's last tactile frame. At rollout time, `Gr00tPolicy` accepts either the complete two-frame pair or a current-only tactile frame. It caches the current frame, repeats it immediately after `reset()`, and uses it as the previous frame on the next call.

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
tactile.rgb           [2, 240, 320, 3] uint8
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

The unchanged RGB-only baseline remains available for ablations:

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

## Closed-loop rollout evaluation

For copy-paste commands with the correct working directory for both terminals, see [CLOSED_LOOP_ROLLOUT.md](CLOSED_LOOP_ROLLOUT.md).

The offline diagnostic above compares dataset actions with predictions; model actions never affect its next input. The closed-loop evaluator instead runs the decoded action through UniVTAC's official `task.take_action(..., action_type="qpos")` API, observes the resulting simulator state, and reports the task's own `check_success()` / `eval_success` result. Its primary metric is per-task Success Rate and the unweighted macro-average across tasks.

The model and simulator stay in separate processes:

```text
UniVTAC / Isaac Sim -> lightweight PolicyClient -> ZeroMQ -> GR00T server
        ^                                                    |
        +---------------- decoded 8-D qpos chunk ------------+
```

All checkpoint preprocessing, action decoding, and unnormalization remain inside `Gr00tPolicy`. The UniVTAC adapter only validates and batches `head`/`wrist` uint8 RGB, the first eight float32 joint coordinates, and `task.instruction`. It executes one predicted action by default, then replans from the new observation. `--execution-horizon` can be set to 1, 4, 8, or 16.

The Isaac environment needs only the client dependencies, not the GR00T package:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install msgpack pyzmq
```

Run `isaaclab.sh` from a clean shell with incompatible Conda environments deactivated; the Isaac Sim 6.0 installation uses its own Python 3.12 runtime.

### 1. Start the GR00T inference server

Run this in the GR00T environment from the GR00T repository root. The inspected local checkpoints are `checkpoint-2500`, `checkpoint-3000`, and `checkpoint-3500`:

```bash
cd /home/yeongyoo/07_Remote/splserver/secret-project-florence

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_gr00t_univtac_server.sh \
  --checkpoint /home/yeongyoo/07_Remote/splserver/outputs/univtac_clean_stats/checkpoint-3500 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

Keep this process running. The evaluator's `--checkpoint` value is recorded in every result row and must describe the checkpoint loaded by this server; the simulator process does not load model weights itself.

### 2. Run the real Isaac Sim smoke evaluation

Run the wrapper from this repository while using the UniVTAC/Isaac Python command. `CUDA_VISIBLE_DEVICES=7` is remapped to the worker's logical `cuda:0`:

```bash
cd /home/yeongyoo/07_Remote/splserver/secret-project-florence

UNIVTAC_ROOT=/path/to/UniVTAC \
UNIVTAC_DRIVER_PYTHON_COMMAND="/home/yeongyoo/IsaacLab/isaaclab.sh -p" \
UNIVTAC_PYTHON_COMMAND="/home/yeongyoo/IsaacLab/isaaclab.sh -p" \
CUDA_VISIBLE_DEVICES=7 \
bash scripts/run_gr00t_univtac_rollout.sh \
  --checkpoint /home/yeongyoo/07_Remote/splserver/outputs/univtac_clean_stats/checkpoint-3500 \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --mode smoke \
  --episodes-per-task 1 \
  --tasks lift_can \
  --execution-horizon 1 \
  --device cuda:0 \
  --output-dir outputs/univtac_rollout_eval_smoke
```

The first rollout prints the exact observation and action shapes/dtypes. It also verifies server reachability, the embedded checkpoint modality config, finite 8-D actions, the simulator's articulation joint limits, multiple closed-loop observations, UniVTAC termination, video finalization, and result-file creation. Omit `--episodes-per-task 1` to use the smoke default of five episodes.

### 3. Run all tasks

The presets are `smoke=5`, `quick=20`, and `full=100` episodes per task. An explicit `--episodes-per-task` always overrides the preset.

```bash
cd /home/yeongyoo/07_Remote/splserver/secret-project-florence

UNIVTAC_ROOT=/path/to/UniVTAC \
UNIVTAC_DRIVER_PYTHON_COMMAND="/home/yeongyoo/IsaacLab/isaaclab.sh -p" \
UNIVTAC_PYTHON_COMMAND="/home/yeongyoo/IsaacLab/isaaclab.sh -p" \
CUDA_VISIBLE_DEVICES=7 \
bash scripts/run_gr00t_univtac_rollout.sh \
  --checkpoint /home/yeongyoo/07_Remote/splserver/outputs/univtac_clean_stats/checkpoint-3500 \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --mode full \
  --tasks all \
  --execution-horizon 1 \
  --device cuda:0 \
  --output-dir outputs/univtac_rollout_eval
```

Selected canonical tasks can be passed after `--tasks`, for example `--tasks lift_can insert_HDMI`. Every task runs in a fresh Isaac process so simulator teardown does not leak state between task classes. Seeds default to `1000000 + episode_index` for every task and are recorded per rollout; use `--seed-start` to change the deterministic sequence. The evaluator refuses to mix a new run with existing result files, so use a new output directory for each checkpoint or configuration.

### 4. Inspect results and videos

```text
outputs/univtac_rollout_eval/
├── episodes.csv
├── summary.csv
├── summary.json
├── summary.md
├── summary.txt
├── logs/
└── videos/
    ├── success/<task>/<task>_seed_<seed>_success.mp4
    ├── failure/<task>/<task>_seed_<seed>_failure.mp4
    └── error/<task>/<task>_seed_<seed>_error.mp4
```

`episodes.csv` records the task, seed, episode index, success, policy action count, execution horizon, checkpoint, instruction, elapsed time, termination reason, video path, simulator-step count, and any error. `summary.csv` and `summary.json` include counts and mean step statistics. The `Average` row is the macro-average of task Success Rates. If evaluated counts differ, an `Overall / Micro Average` is also emitted. `exception` and `invalid_action` rollouts are explicit errors, not silently counted as policy failures or included in the Success Rate denominator.

## Common failures

### Sparsh-DINO tactile representation

The default multimodal tactile backend is `sparsh_dino_base`; `resnet18` remains available as the scratch ablation. The implementation follows Meta's [official Sparsh repository](https://github.com/facebookresearch/sparsh) and the [official model card](https://huggingface.co/facebook/sparsh-dino-base): ViT-B/16, 768 hidden dimensions, one register token, and 300 returned normalized patch tokens at 320×240. The GR00T adapter is `LayerNorm(768) -> Linear(768, DiT dim)` and keeps all patch tokens.

Official preprocessing is applied once inside `SparshDinoTactileEncoder`: RGB channel order, uint8-to-`[0,1]`, landscape-to-portrait rotation, 4:3 center crop, antialiased resize to `(H,W)=(320,240)`, then `I_t ⊕ I_previous`. There is no ImageNet mean/std normalization. The processor only converts HWC to CHW and scales uint8; it does not resize or normalize again. DIGIT and GelSight Mini pretraining used sensor-specific no-contact background subtraction. Supply that calibration image with `--tactile-background-path /path/to/no_contact.png`; without it the wrapper logs an explicit warning and retains raw RGB, matching the official GelSight-2017 path but not GS Mini background-subtracted preprocessing.

Download the exact safe official backbone file when a local checkpoint is preferred:

```bash
uv run hf download facebook/sparsh-dino-base dino_vitbase.safetensors \
  --local-dir checkpoints/sparsh-dino-base
```

Use either `--tactile-pretrained-model facebook/sparsh-dino-base` or the resulting local directory/file. Loading is strict and fails on missing or unexpected keys; it never falls back to random Sparsh weights. Fine-tuned GR00T checkpoints contain the full Sparsh backbone, projection, tactile cross-attention, gates, and optional background calibration tensor. Their saved config disables the original Hub bootstrap, so inference reloads the embedded weights without redownloading Sparsh or requiring the original background file. Meta's extracted Sparsh backbone implementation in `gr00t/model/extension/sparsh_vit.py` and the official model artifact are licensed CC-BY-NC-4.0; review that license independently of GR00T's Apache-2.0 code.

### Architecture

```text
Head RGB ───────────────┐
Wrist RGB ───────────────┼→ Qwen3-VL ─────────────┐
Language ───────────────┘                         │
                                                  ↓
                                            GR00T Action DiT
                                           ↑               ↑
                                          /                 \
                           point cross-attn                   tactile cross-attn
                                ↑                                  ↑
                         Concerto pretrained                 Sparsh-DINO pretrained
                                ↑                                  ↑
                      Head+Wrist scene PC                Tactile(t-1), Tactile(t)
```

The Concerto and Sparsh branches stay independent until their separate gated cross-attention residuals update action tokens. Tactile images are not sent through Qwen3-VL.

### Multimodal fine-tuning

This command parses through `examples/finetune.sh`, uses LR `1e-5` only for the pretrained Sparsh backbone, and uses the action-head LR (`1e-4`) for the new projection, cross-attention, and gates:

```bash
CUDA_VISIBLE_DEVICES=0,7 \
NUM_GPUS=2 \
GLOBAL_BATCH_SIZE=2 \
DATALOADER_NUM_WORKERS=0 \
EPISODE_SAMPLING_RATE=1.0 \
MAX_STEPS=2000 \
USE_WANDB=0 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/univtac_multimodal \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UniVTAC/univtac_multimodal_config.py \
  --output-dir outputs/univtac_sparsh \
  -- \
  --dit-type multimodal_conditioned_dit \
  --point-input-dim 6 \
  --tactile-encoder-cfg sparsh_dino_base \
  --tactile-pretrained-model facebook/sparsh-dino-base \
  --tactile-background-path /path/to/gsmini_no_contact.png \
  --tactile-encoder-learning-rate 1e-5 \
  --action-head-learning-rate 1e-4 \
  --allow-padding \
  --tune-tactile-encoder \
  --tune-multimodal-adapter
```

`--allow-padding` uses GR00T's established in-episode clamping for all delta-indexed modalities;
therefore the first tactile pair is `[frame 0, frame 0]`, matching rollout reset behavior. The
background argument reproduces the official GelSight Mini/DIGIT subtraction. Omit it only
when the stored stream is already background-adjusted or when intentionally using the official
raw GelSight-2017 preprocessing path.

Use `--no-tune-tactile-encoder` to freeze only the Sparsh backbone; the newly initialized tactile projection and cross-attention remain trainable when `--tune-multimodal-adapter` is enabled. Use `--tactile-encoder-cfg resnet18` for the scratch baseline.

One-batch real-checkpoint CUDA smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test_sparsh_tactile.py \
  --pretrained-model checkpoints/sparsh-dino-base \
  --batch-size 1 \
  --dit-dim 1536
```

Use `--no-use-point-conditioning` or `--no-use-tactile-conditioning` for a single-sensor ablation. The calibrated UniVTAC `scene` conversion is XYZRGB and uses the default `--point-input-dim 6`; legacy XYZ-only point fields must use a lightweight point encoder with `--point-input-dim 3`.

## Current Limitations

The extension includes pretrained Sparsh-DINO-Base plus a scratch ResNet-18 ablation, Concerto/lightweight point-encoder choices, and gated action-token cross-attention. It does **not** fuse point and tactile representations before the action DiT. Point coordinates remain in the selected depth camera frame and receive no dataset-statistics normalization.

## Backward Compatibility

Existing RGB + state + action + language datasets do not need `tactile`, `pointcloud`, or `pointcloud_path` metadata. Conversion without `--include-tactile` and `--include-pointcloud` produces the prior layout. Continue using [univtac_config.py](univtac_config.py) with the default `alternate_vl_dit`; only use [univtac_multimodal_config.py](univtac_multimodal_config.py) with matching converted modalities and `--dit-type multimodal_conditioned_dit`.

Common failures remain fail-fast: missing head/wrist/tactile streams, temporal-length mismatches, odd H.264 dimensions, invalid/missing depth calibration, a frame with no valid depth, or a point-cloud NPZ whose row count differs from the episode DataFrame.
