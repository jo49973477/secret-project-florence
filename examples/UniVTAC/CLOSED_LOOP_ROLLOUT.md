# GR00T + UniVTAC closed-loop rollout guide

This guide uses two terminals and two Python environments:

- Terminal 1 runs the GR00T inference server from the GR00T repository.
- Terminal 2 launches UniVTAC/Isaac Sim through Isaac Lab.

The rollout wrapper itself lives in the GR00T repository, so both commands below start from `GR00T_ROOT`. The worker automatically changes its working directory to `UNIVTAC_ROOT` before importing UniVTAC. Do not copy GR00T into the UniVTAC environment or install both projects into one Python environment.

## 0. Set and verify paths

Open a clean shell. Deactivate incompatible Conda environments before invoking Isaac Lab; Isaac Sim 6.0 uses its own Python 3.12 runtime.

```bash
GR00T_ROOT=/home/yeongyoo/07_Remote/splserver/secret-project-florence
UNIVTAC_ROOT=/absolute/path/to/UniVTAC
ISAACLAB_ROOT=/home/yeongyoo/IsaacLab
CHECKPOINT=/home/yeongyoo/07_Remote/splserver/outputs/univtac_clean_stats/checkpoint-3500

test -f "$GR00T_ROOT/gr00t/eval/run_gr00t_server.py"
test -f "$UNIVTAC_ROOT/envs/_base_task.py"
test -f "$UNIVTAC_ROOT/scripts/eval_policy.py"
test -x "$ISAACLAB_ROOT/isaaclab.sh"
test -d "$CHECKPOINT"
```

Replace only `UNIVTAC_ROOT` if your UniVTAC checkout is elsewhere. The inspected local checkpoints are `checkpoint-2500`, `checkpoint-3000`, and `checkpoint-3500` under `outputs/univtac_clean_stats/`.

Install the lightweight client dependencies into the Isaac environment once:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install msgpack pyzmq
```

## 1. Terminal 1: start the GR00T server

Use the GR00T environment and run from the GR00T repository root:

```bash
cd "$GR00T_ROOT"

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_gr00t_univtac_server.sh \
  --checkpoint "$CHECKPOINT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

Wait for this message and leave the terminal running:

```text
Server ready — listening on 127.0.0.1:5555
```

`CUDA_VISIBLE_DEVICES=0` selects physical GPU 0. Inside the process that GPU is addressed as `cuda:0`.

## 2. Terminal 2: run the one-episode smoke test

Open another clean shell, set the same variables again, and run the wrapper from `GR00T_ROOT`:

```bash
GR00T_ROOT=/home/yeongyoo/07_Remote/splserver/secret-project-florence
UNIVTAC_ROOT=/absolute/path/to/UniVTAC
ISAACLAB_ROOT=/home/yeongyoo/IsaacLab
CHECKPOINT=/home/yeongyoo/07_Remote/splserver/outputs/univtac_clean_stats/checkpoint-3500

cd "$GR00T_ROOT"

UNIVTAC_ROOT="$UNIVTAC_ROOT" \
UNIVTAC_DRIVER_PYTHON_COMMAND="$ISAACLAB_ROOT/isaaclab.sh -p" \
UNIVTAC_PYTHON_COMMAND="$ISAACLAB_ROOT/isaaclab.sh -p" \
CUDA_VISIBLE_DEVICES=7 \
bash scripts/run_gr00t_univtac_rollout.sh \
  --checkpoint "$CHECKPOINT" \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --mode smoke \
  --episodes-per-task 1 \
  --tasks lift_can \
  --execution-horizon 1 \
  --device cuda:0 \
  --output-dir "$GR00T_ROOT/outputs/univtac_rollout_eval_smoke"
```

`CUDA_VISIBLE_DEVICES=7` selects physical GPU 7 for Isaac Sim. The worker sees that remapped GPU as `cuda:0`, which is why `--device cuda:0` is correct.

The smoke test should show:

1. GR00T server connectivity and checkpoint-modality validation.
2. `head` and `wrist` arrays shaped `[1, 1, H, W, 3]` with `uint8` dtype.
3. Joint state shaped `[1, 1, 8]` with `float32` dtype.
4. A finite predicted action chunk shaped `[horizon, 8]`.
5. At least one qpos action executed by UniVTAC.
6. A subsequent simulator observation, normal termination, a finalized video, and result files.

## 3. Inspect the smoke result

Run from `GR00T_ROOT`:

```bash
cd "$GR00T_ROOT"

cat outputs/univtac_rollout_eval_smoke/summary.txt
column -s, -t outputs/univtac_rollout_eval_smoke/episodes.csv | less -S
find outputs/univtac_rollout_eval_smoke/videos -type f -name '*.mp4' -print
```

The output directory contains:

```text
outputs/univtac_rollout_eval_smoke/
├── episodes.csv
├── summary.csv
├── summary.json
├── summary.md
├── summary.txt
├── logs/
└── videos/
    ├── success/<task>/
    ├── failure/<task>/
    └── error/<task>/
```

Use a new output directory for every run. The evaluator intentionally refuses to mix new results with an existing `episodes.csv` or `summary.json`.

## 4. Run all eight tasks

After the smoke test succeeds, run the full evaluation from `GR00T_ROOT`:

```bash
cd "$GR00T_ROOT"

UNIVTAC_ROOT="$UNIVTAC_ROOT" \
UNIVTAC_DRIVER_PYTHON_COMMAND="$ISAACLAB_ROOT/isaaclab.sh -p" \
UNIVTAC_PYTHON_COMMAND="$ISAACLAB_ROOT/isaaclab.sh -p" \
CUDA_VISIBLE_DEVICES=7 \
bash scripts/run_gr00t_univtac_rollout.sh \
  --checkpoint "$CHECKPOINT" \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --mode full \
  --tasks all \
  --execution-horizon 1 \
  --device cuda:0 \
  --output-dir "$GR00T_ROOT/outputs/univtac_rollout_eval_checkpoint-3500"
```

Evaluation modes are:

| Mode | Episodes per task |
|---|---:|
| `smoke` | 5 |
| `quick` | 20 |
| `full` | 100 |

Use `--episodes-per-task N` to override the selected mode. Use one or more canonical task names after `--tasks` to evaluate a subset.

## Important checkpoint rule

The server's `--checkpoint` selects the model weights. The rollout evaluator's `--checkpoint` records the checkpoint in `episodes.csv`, summaries, and logs. Pass the same path to both terminals so the report accurately identifies the model that produced the actions.

## Open-loop versus closed-loop

- Open-loop diagnostics compare predicted actions with dataset ground-truth actions without changing the next observation.
- Closed-loop rollout evaluation executes model actions in UniVTAC, observes the resulting simulator state, and reports UniVTAC task Success Rate.
