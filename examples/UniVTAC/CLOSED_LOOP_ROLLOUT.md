# GR00T + UniVTAC closed-loop rollout guide

The GR00T server and UniVTAC simulator intentionally run in separate Python environments:

- The SPL server runs `gr00t/eval/run_gr00t_server.py` and owns the checkpoint.
- The local machine runs UniVTAC and Isaac Sim and reaches the server through the SSH tunnel.
- The rollout is a real observation → inference → qpos action → simulator step → new observation loop.

## Supported simulator combinations

Use the environment that matches the checked-out UniVTAC code and task YAML. The worker validates this before starting Kit.

| UniVTAC code/config | Isaac Sim | Isaac Lab | Python |
|---|---:|---:|---:|
| `main` (flat `observations` config) | 4.5.x | release v2.1.1 (package 0.41.3) | 3.10 |
| `isaac51` (nested `env_settings` config) | 5.1.x | 2.3.0 | 3.11 |

Isaac Sim 6.0 / Isaac Lab 3.0 is not supported by either official UniVTAC branch. UniVTAC also requires its vendored `tacex`, `tacex_assets`, `tacex_uipc`, `uipc`, and pinned cuRobo modules. Keep any compatible UniVTAC environment isolated; do not downgrade or overwrite an existing Isaac Sim 6.0 installation.

The current public UniVTAC policy checkpoints and Isaac 4.5 demonstrations are associated with `main`. When evaluating a GR00T checkpoint trained from that data, prefer an isolated `main` + Isaac Sim 4.5 + Isaac Lab 2.1.1 environment unless the training data was explicitly produced by `isaac51`.

## 1. SPL server: start GR00T (Bash)

```bash
cd /home/yeongyoo/07_Remote/splserver/secret-project-florence

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_gr00t_univtac_server.sh \
  --checkpoint /ssdg/spl_yeongyoo/outputs/univtac_clean_stats/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

Leave the server running. Establish the existing SSH forward so local `127.0.0.1:5555` reaches SPL `127.0.0.1:5555`.

## 2. Local machine: one-episode smoke rollout (Fish)

The official `main` installer creates the Conda environment `UniVTAC` and clones its pinned Isaac Lab below the UniVTAC checkout. The commands below resolve that environment without activating it. They deliberately fail before launch if the isolated environment has not been installed; they never fall back to Isaac Sim 6.0.

```fish
set -gx GR00T_ROOT /home/yeongyoo/07_Remote/splserver/secret-project-florence
set -gx UNIVTAC_ROOT /home/yeongyoo/03_Dataset/07_UnivTac/UniVTAC
set -gx ISAACLAB_ROOT $UNIVTAC_ROOT/third_party/IsaacLab
set -gx CONDA_BASE (conda info --base)
set -gx ISAAC_PYTHON $CONDA_BASE/envs/UniVTAC/bin/python
set -gx ISAAC_LAUNCHER $ISAAC_PYTHON
set -gx CHECKPOINT /ssdg/spl_yeongyoo/outputs/univtac_clean_stats/checkpoint-10000
set -gx OUT /home/yeongyoo/univtac_rollout_10000_smoke_(date +%Y%m%d_%H%M%S)

cd $GR00T_ROOT

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV -u VIRTUAL_ENV_PROMPT \
  UNIVTAC_ROOT=$UNIVTAC_ROOT \
  UNIVTAC_EXPECTED_PYTHON=$ISAAC_PYTHON \
  UNIVTAC_DRIVER_PYTHON_COMMAND=$ISAAC_LAUNCHER \
  UNIVTAC_PYTHON_COMMAND=$ISAAC_LAUNCHER \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_gr00t_univtac_rollout.sh \
    --checkpoint $CHECKPOINT \
    --server-host 127.0.0.1 \
    --server-port 5555 \
    --mode smoke \
    --episodes-per-task 1 \
    --tasks lift_can \
    --execution-horizon 1 \
    --device cuda:0 \
    --output-dir $OUT
```

The exact-interpreter check prevents an active Conda environment, virtualenv, or system Python from being selected accidentally.

## 3. Verify the result (Fish)

```fish
cat $OUT/summary.txt
cat $OUT/episodes.csv
find $OUT/videos -type f -name '*.mp4' -print
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,duration \
  -of default=noprint_wrappers=1 (find $OUT/videos -type f -name '*.mp4' | head -n 1)
```

A valid one-step smoke run logs all of the following:

1. GR00T ping and checkpoint modality validation.
2. Task construction and reset completion.
3. Head/wrist RGB shapes, the 8-dimensional joint state, and predicted `[horizon, 8]` action.
4. The exact 8-dimensional qpos command passed to `task.take_action`.
5. A finite post-action joint observation.
6. FFmpeg PID, frame count, exit code, finalized path, and readable H.264 MP4 validation.

## Failure semantics

Worker output is unbuffered and each lifecycle phase is flushed to `logs/<task>.log`. The log footer records the worker PID, return code, signal (when the OS exposes one), and elapsed time.

`logs/worker_results/<task>.json` records `phase`, `completed`, `fatal_error`, episode rows, and runtime metadata. The orchestrator requires the worker to explicitly report completion and return exactly the requested episode indices. A crash, missing result file, malformed result, nonzero exit, or zero-episode return creates explicit `worker_failure` rows and an `incomplete` summary before the command exits nonzero. Such rows never claim a video or successful evaluation.

Every completed success/failure episode must finalize an actual MP4. The worker checks that FFmpeg started, received frames, exited successfully, produced a nonempty file, and that `ffprobe` (or OpenCV fallback) can read its video stream. A video failure converts the episode to an error rather than reporting a successful or failed rollout without an artifact.

Use a new output directory for each run; the evaluator refuses to mix results from different runs.
