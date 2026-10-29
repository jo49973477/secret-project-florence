# Hydra experiment configuration

Hydra is the experiment composition interface. The legacy Tyro launchers and shell
workflows remain available. No neural-network equations, losses, optimizer ownership
rules, checkpoint formats, or training datasets were changed by this migration.
The pre-implementation [inventory](HYDRA_CONFIGURATION_INVENTORY.md) documents the legacy
CLI fields and effective precedence, including differences already present in the working tree.

```text
configs/
├── model/{gr00t_base,gr00t_multimodal}.yaml
├── data/{univtac_rgb,univtac_multimodal,so100}.yaml
├── training/{finetune,overfit_5eps,debug}.yaml
├── hardware/{single_3090,dual_3090_zero3}.yaml
└── experiment/{univtac_full,univtac_rgb,univtac_overfit5,so100_baseline}.yaml
```

`gr00t/configs/hydra_config.py` converts a resolved DictConfig directly into
`Config(model=Gr00tN1d7Config(...), data=DataConfig(...), training=TrainingConfig(...))`.
It never instantiates FinetuneConfig. Existing dataclass field types supply the
OmegaConf validation schemas; there is no independent copy of the internal schema.
Unknown keys, including keys introduced with `+`, fail conversion. Small additional
schemas describe the experiment, hardware launch options and resume selection.
No YAML executes Python and no custom resolvers are needed.

## Ownership and precedence

1. Hydra composes group defaults, then experiment settings, then CLI overrides.
   Group variants inherit their common recipe; `_self_` sets intentional differences.
2. Model tuning, conditioning, supported encoder options and dropout overrides map to
   the exact fields used by the existing `setup.py` loader. `model.base_model_path`
   maps to `training.start_from_checkpoint`. Data preprocessing maps to processor-facing
   model fields; it has one experiment-level home under `data.preprocessing`.
   Hardware training fields map to TrainingConfig. The root output_dir maps to
   TrainingConfig.output_dir; experiment_name is left None to prevent a second nesting.
3. Core architecture belongs to the checkpoint: layer counts, widths, diffusion config,
   state/action dimensions and maximum horizon. Arbitrary architectural overrides are
   rejected. `model.checkpoint_expectations` contains **assertions**, never instructions
   to resize a checkpoint. The default asserts maximum action horizon 40. Workers read
   checkpoint metadata, check assertions and embedded sensor/LoRA compatibility, and
   copy checkpoint-owned dimensions into the internal config before training.
4. `setup.py` still performs AutoModel loading, its explicit tuning/sensor/dropout
   overrides, strict weight mismatch checks, embedded sensor validation and deferred
   encoder bootstrap. It inherits checkpoint architecture instead of replacing it with
   the entire internal dataclass. LoRA enablement/loading follows the existing path.
5. Processor settings and dataset modalities come from Hydra through the existing loader.
   Processor state dropout is zero; the action head applies the configured probability
   once. `final_model_config.json` and `final_processor_config.json` record loaded values.
6. Full resume restores model, optimizer, scheduler, RNG and Trainer step state through
   the unchanged Trainer/DeepSpeed path. Restored current LRs take precedence over initial
   optimizer values. Requested LR changes are rejected during resume. Constructor defaults
   for internal fields without experiment controls remain owned by the existing dataclasses
   and are included in the internal config snapshot. They are not copied into parallel defaults.

The sharded DatasetFactory only supports single_turn and eval_strategy=no. Existing
unused legacy fields (eval split ratio, eval_bf16, shuffle/subsample/random_chop, upload,
RL and built-in open-loop toggles) are rejected by this interface. Use separate open-loop
evaluation commands. Dataset selection/mixing, sharding, episode sampling, padding,
preprocessing, optimizer/scheduler, clipping, warmup, tracking and notifications are configurable.
Internal data loading shuffle behavior and `ignore_data_skip=True` remain unchanged.
Resume therefore does **not** promise exact replay of the sample stream.

## Running and inspecting experiments

Install the project environment with `uv sync --all-extras`. The Hydra dependency is
pinned to 1.3.2 in the main and platform manifests and recorded in uv.lock.

Dataset defaults are portable placeholders, not machine-specific absolute paths.
Point FLORENCE_DATASET at an **existing converted** UniVTAC dataset. The launcher does
not convert data or create subsets. The SO100 recipe uses the existing
`demo_data/cube_to_bowl_5` and `examples/SO100/so100_config.py` modalities.

```bash
# Default multimodal experiment; select GPU IDs through the environment or YAML override.
FLORENCE_DATASET=/path/to/univtac_full CUDA_VISIBLE_DEVICES=0,1 \
uv run python scripts/train_hydra.py \
    --config-name experiment/univtac_full

# Change training hyperparameters.
uv run python scripts/train_hydra.py \
    --config-name experiment/univtac_full \
    training.max_steps=2000 \
    training.action_head_learning_rate=5e-5

# Change hardware configuration (clear or update a conflicting CUDA_VISIBLE_DEVICES value).
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_hydra.py \
    --config-name experiment/univtac_overfit5 \
    hardware=single_3090

# Inspect composed configuration without importing the conversion or training code.
uv run python scripts/train_hydra.py \
    --config-name experiment/univtac_full \
    --cfg job --resolve

# Validated preflight: no checkpoint reads, downloads, model, CUDA or distributed init.
uv run python scripts/train_hydra.py \
    --config-name experiment/univtac_full dry_run=true

# Existing RGB-only modalities (16 dataset steps, maximum 40).
uv run python scripts/train_hydra.py --config-name experiment/univtac_rgb dry_run=true

# Existing SO100 recipe: front/wrist, relative single_arm and absolute gripper.
uv run python scripts/train_hydra.py --config-name experiment/so100_baseline dry_run=true
```

Dry-run prints resolved settings, selected IDs, world size, microbatch and accumulated
batch size, requested LR policy and conditioning. It validates configuration compatibility
but intentionally does not check checkpoint files or availability of physical GPUs. An
asserted checkpoint maximum remains unverified until worker startup. Dataset existence and
an asserted five-episode subset are checked on a real launch. A preflight is not a GPU memory
estimate or proof that the job fits a 3090.

Change a dataset with a Hydra list override (quote it for the shell):

```bash
uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  'data.datasets=[{dataset_paths: [/path/to/converted_dataset], embodiment_tag: new_embodiment, mix_ratio: 1.0}]' \
  dry_run=true
```

A dataset action horizon can be shorter than the checkpoint maximum. The multimodal YAML
uses horizon **40** as requested; the current working-tree Python builder uses **16**.
To match it, set `data.modalities.action.horizon=16`. Horizon above the model maximum fails.
UniVTAC requires head/wrist video, joint state/action, the task description, tactile.rgb
with RGB channels, and pointcloud.scene with six XYZRGB features. Sparsh-DINO requires
exactly two chronological offsets `[negative_offset, 0]`, default `[-1, 0]`.
No dimensions are silently repaired. Missing sensor modalities fail when their conditioning
flags are enabled. Disable a conditioning flag to ablate that branch while retaining the
available dataset modality; the DiT must remain multimodal if any sensor is enabled.

```bash
# Supported lightweight encoder alternatives; no sensor checkpoint downloaded in preflight.
uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  model.point_encoder_cfg=pointnet2 model.tactile_encoder_cfg=resnet18 dry_run=true

# Independent backbone LRs; sensor projections/adapters stay in the action-head groups.
uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  training.vlm_learning_rate=1e-4 training.action_head_learning_rate=1e-3 \
  training.point_encoder_learning_rate=1e-5 training.tactile_encoder_learning_rate=1e-5 \
  dry_run=true
```

Unset VLM/head LR falls back to learning_rate. Unset sensor backbone LR falls back to head
LR, matching TrainingConfig semantics. The full recipe defaults all effective LRs to 1e-4,
matching the full shell recipe, rather than the raw FinetuneConfig sensor default 1e-5.
Frozen or absent backbones have no trainable optimizer group. Actual group ownership,
decay, tensor counts and parameter counts come from Trainer, not a preflight estimate.

## Distributed execution and interruption

Run `train_hydra.py` **once**, not under Torchrun. It writes a resolved snapshot, sets
CUDA_VISIBLE_DEVICES and launches an argument-array Torchrun command. Even a single GPU
uses one Torchrun worker so the existing distributed initialization contract is preserved.
Every rank reads the same absolute snapshot; workers never compose Hydra or choose run IDs.
GPU count must match distinct selected IDs, and the pre-accumulation global batch must be
divisible by world size. Effective optimizer batch is global_batch_size × accumulation.
DeepSpeed stage 2 or 3 remains selectable with `hardware.deepspeed_stage=2` or `3`;
single-GPU training follows the existing non-DeepSpeed path. ZeRO3 memory controls retain
existing validation and require multiple GPUs. DDP remains an existing alternative.

The environment inherits platform activation helpers and caller CUDA/PyTorch settings.
Allocator, Albumentations update suppression and log level get legacy-compatible defaults
only when absent. Torchrun supervision preserves Discord failure notifications; Trainer
retains W&B, Telegram and Discord behavior. Secrets stay in their existing environment-only
transports. Termination is forwarded to the process group; child failure codes are returned.
Output is streamed to the terminal and logs/train.log.

## Fresh fine-tuning and full resume

Fresh fine-tuning uses model.base_model_path as a pretrained or previously saved,
Transformers-loadable model checkpoint. A model-only checkpoint may be a fresh fine-tuning
source. It is not a full resume source. Native ZeRO partitions require the existing export
workflow for standalone inference/fine-tuning; they are restored directly only by DeepSpeed.

Fresh output directories must be empty. Resume must explicitly select the original output
directory; it never creates a replacement timestamped training directory.

```bash
# Latest valid checkpoint; ignores incomplete newer checkpoints.
uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  output_dir=/absolute/path/to/original_run \
  resume.enabled=true resume.auto_find_latest=true training.max_steps=7000

# Explicit checkpoint; keep original dataset, model, batch, precision and optimizer settings.
uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  output_dir=/absolute/path/to/original_run resume.enabled=true \
  resume.checkpoint_path=/absolute/path/to/original_run/checkpoint-4000 \
  training.max_steps=7000
```

Validation requires trainer_state.json with matching step, optimizer, scheduler, RNG and
model weights/partitions. DeepSpeed optimizer/model state patterns match the shell helper.
This checks state families without unpickling files; DeepSpeed/Trainer still perform the
actual restoration and detect corrupt/incomplete partition contents. Model/data settings,
optimizer/LR policy, accumulation, global batch, precision and world size must match saved
Hydra provenance. A single process count is not enough to prove the contents of all ZeRO
shards; GPU restoration still needs integration testing.

Operational changes allowed: increasing max_steps, logging/saving cadence and retention,
notification/tracking settings, selected physical IDs, port and loader workers. Extending
max_steps uses the existing HF scheduler construction plus restored scheduler state; it
is not a guarantee of an identical LR curve to a run planned for the longer duration.
Initial model construction still uses the original base, then Trainer restores full state,
which preserves native ZeRO and sensor-bootstrap behavior. Keep the original base/sensor
assets available. The old experiment snapshot is retained, and resume attempts receive
separate `.hydra/resume_<timestamp>/` manifests. Current effective optimizer LRs are recorded
at on_train_begin, after restoration; historical manifests stay in their resume directories.

For legacy runs without `.hydra/config.yaml`, full compatibility cannot be verified.
Use the existing shell/Tyro resume path. Hydra refuses these full resumes, rather than
claiming compatibility from weights alone. To migrate weights, use a fresh output directory
and a loadable model checkpoint as model.base_model_path.

## Five-episode memorization workflow

Preparation and evaluation are separate reusable commands. Preparation selects IDs
`0,1,2,3,4`, copies metadata, symlinks episode files, recomputes normalization statistics
for the new subset and refuses an existing target. It never deletes or rewrites source data.
The overfit recipe asserts that the supplied subset contains exactly those five IDs.
It disables jitter, state/head/self-attention dropout and weight decay, uses crop fraction 1,
full min/max statistics, LR 3e-4 and no accumulation, retaining 50 checkpoints at 5000 steps.

```bash
uv run python scripts/univtac_overfit_workflow.py prepare \
  --source /path/to/univtac --target /path/to/new_univtac_overfit5

FLORENCE_OVERFIT_DATASET=/path/to/new_univtac_overfit5 CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/train_hydra.py --config-name experiment/univtac_overfit5 \
  hardware=single_3090 output_dir=outputs/univtac_memorization

uv run python scripts/univtac_overfit_workflow.py evaluate \
  --dataset /path/to/new_univtac_overfit5 --output outputs/univtac_memorization --gpu 0
```

Evaluation runs the existing open_loop_eval.py on every saved checkpoint and the same five
trajectories, retaining logs and memorization_metrics.csv. This measures memorization,
not generalization. With native ZeRO3 checkpoints, consolidate each selected checkpoint
using `scripts/export_zero3_checkpoint.py CHECKPOINT NEW_EXPORT --model-config-dir HF_MODEL_DIR`
on a host with sufficient RAM, using model metadata matching the trained multimodal model.
Evaluate the exports (place exported directories under an evaluation run directory as
checkpoint-N) with the same wrapper. Export is explicit and is never triggered as a hidden
part of configuration parsing. The old overfit script remains available as a rollback.

## Artifacts and reproduction

```text
outputs/<name>/<run_id>/
├── .hydra/{config.yaml,hydra.yaml,overrides.yaml,gr00t.yaml,metadata.json}
├── .hydra/resume_<timestamp>/...       # only for resume attempts
├── experiment_cfg/{config.yaml,conf.yaml,final_model_config.json,final_processor_config.json,...}
├── effective_optimizers.json
├── metadata.json
├── logs/train.log
└── checkpoint-*/                      # native existing Trainer checkpoint format
```

The parent owns `.hydra` and metadata; existing rank-zero writes own experiment_cfg and
processor/checkpoint artifacts. Hydra job logging/output snapshots are disabled at its
bootstrap directory `.`; the parent writes them only after validation, at the final stable
output_dir. Thus neither failed preflight nor a resume makes a stray timestamped Hydra run.
The small gr00t.yaml snapshot records all internal defaults before loading. Existing
experiment_cfg records worker architecture after metadata inheritance and final loaded
model/processor configs; no weights are duplicated by the config launcher.

metadata.json contains commit hash and dirty flag, experiment/run IDs and timestamp,
selected groups, CLI overrides, dataset paths, base model, batch/world size, accumulation,
precision, conditioning and resume source. The resolved Hydra config is also sent to W&B
as hydra_resolved alongside the existing internal config. Secret environment variables are
not enumerated or serialized. Inline Transformers tokens are rejected; use standard Hub
environment authentication. Keep credentials out of paths and CLI overrides.

For reproduction use the same commit and code changes, dataset content, model revision
and external sensor assets. Dataset/weight content hashes are not computed. A dirty flag
is evidence of local modifications, not a backup of them. Pin a revision for Hub assets
through `+model.model_revision=...` (the existing loading path applies it). Reproduce
from a saved resolved snapshot and select a new output directory:

```bash
uv run python scripts/train_hydra.py \
  --config-path /absolute/path/to/original_run/.hydra --config-name config \
  output_dir=outputs/reproduction run_id=reproduction \
  resume.enabled=false resume.checkpoint_path=null resume.auto_find_latest=false \
  hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled
```

To add a recipe, create a group YAML inheriting a current variant and select it from a
new `experiment/NAME.yaml` with `# @package _global_`. Keep architectural assertions in
checkpoint_expectations, modality objects in data.modalities, preprocessing in
data.preprocessing, hardware controls in hardware and optimizer policy in training.
Use horizon integers and enum names such as RELATIVE/NON_EEF/DEFAULT for actions.
Run `--cfg job --resolve`, `dry_run=true` and CPU config tests before a GPU job.
To expose a new internal field, first verify that its runtime consumer actually uses it,
then update the explicit supported-field mapping and tests. Unknown/unused fields fail.

## Validation and boundaries

The CPU suites cover composition, overrides, typed conversion, modality parity, horizons,
sensor contracts, GPU/batch checks, LR fallbacks, checkpoint state families and compatibility,
snapshots, subset preparation safety, rank-zero optimizer reporting, dry-run import guards,
and a mocked subprocess launch with exit-code propagation. They do not establish GPU memory
fit, distributed liveness, DeepSpeed restoration, external encoder download success, or
convergence. Preserve the legacy path until those integration checks pass on target hardware.


Intentional migration differences: RGB recipes explicitly disable inactive sensor flags
(the old alternate DiT ignored their True defaults); portable dataset paths replace
machine-specific shell paths; output naming/retention follows the selected YAML recipe.
The SO100 baseline keeps the common five-checkpoint retention instead of the overfit shell
script's keep-all policy; set training.save_total_limit=20 for its 2000-step/100-save recipe.


Real launches also run an isolated sensor dependency import check once before
Torchrun and checkpoint loading. Configuration-only `dry_run=true` is unchanged.
See [sensor dependency installation and compatibility](sensor_dependencies.md)
for Concerto/SPARSH requirements and Python 3.12 / Torch 2.9.0+cu128 guidance.


Relative action statistics use the resolved per-embodiment Hydra modalities from
`config.data.modality_configs`, including custom `new_embodiment` definitions.
`generate_rel_stats(..., modality_configs=resolved_modalities)` passes those same
settings to the relative-action loader and cache fingerprint. Omitting the
argument retains the built-in registry fallback. Existing fingerprint payloads
and atomic cache writes are unchanged; changed horizons or action semantics
regenerate incompatible entries. UniVTAC retains relative `joint` actions
against state `joint` (`NON_EEF`, `DEFAULT`, `state_key=joint`), with the RGB
recipe's 16-step and multimodal recipe's 40-step data horizons.

Experiment identity, W&B grouping and hardware batch defaults are documented in
[experiment_identity.md](experiment_identity.md).
