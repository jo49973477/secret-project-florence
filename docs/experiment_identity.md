# Hydra experiment identity and batch defaults

`name` is a stable semantic identifier; `run_id` retains `${now:%Y%m%d_%H%M%S}`.
The output remains `outputs/${name}/${run_id}`. Hydra passes a separate runtime
`training.wandb_name` (`${name}-${run_id}`), leaving `training.experiment_name=None`
so the legacy runner does not append another directory. Do not override
`training.wandb_name` in Hydra; override `name` instead.

| Preset | name | W&B group |
| --- | --- | --- |
| experiment/univtac_rgb | base-full | base |
| experiment/univtac_overfit5 | base-overfit5 | base |
| experiment/univtac_full | multimodal-full | multimodal |
| Sidecar A | sidecar-point-full | sidecar |
| Sidecar B | sidecar-contact-warmup | sidecar |
| Sidecar C | sidecar-full | sidecar |
| Sidecar overfit5 | sidecar-full-overfit5 | sidecar |

UniVTAC presets share project `florence-univtac`. W&B remains opt-in through the
existing `training.use_wandb` setting: this change does not enable it in disabled
presets. `training.wandb_group` is optional (`null` omits the init argument).
Future Contact-MoE runs can use `name=contact-moe-full
training.wandb_group=contact-moe`; this adds no architecture preset.

For example:

```bash
bash scripts/train_univtac_sidecar.sh overfit5 name=custom_run training.max_steps=500 dry_run=true
python scripts/train_hydra.py --config-name experiment/univtac_rgb dry_run=true
```

A default overfit execution may produce
`outputs/sidecar-full-overfit5/20261029_193000` and W&B display name
`sidecar-full-overfit5-20261029_193000`, group `sidecar`.
CLI overrides follow wrapper defaults, so explicit `name`, project, group, or
training overrides win. For a literal timestamp override, retain Hydra quotes:
`'run_id="20261029_193000"'` (an unquoted numeric underscore can parse as an integer).

## Batch precedence

Hardware may supply `global_batch_size` and `gradient_accumulation_steps`.
Finetune selects those via `oc.select`, with legacy fallback **32 / 1** for profiles
without them. A training preset may replace either value; overfit explicitly uses
GA=1. Explicit CLI `training.global_batch_size` / `training.gradient_accumulation_steps`
win. Explicit CLI `hardware.*` changes defaults only; it cannot override an explicit
training value. Unsupported hardware keys (including `batch_size`) are rejected.

Global batch is the total across GPUs **before** accumulation. Per-device batch is
`global_batch_size / num_gpus`; effective batch is `global_batch_size * GA`.
Positive, divisible global batch and GA >= 1 are validated before launching workers.
For effective batch 32, use 4 GPUs / global 4 / GA 8 or 2 GPUs / global 2 / GA 16.

Local dual and quad profiles were integrated without changing their supplied
numbers: dual global 4 / GA 8; quad global 32 / GA 1. Their formerly unsupported
`batch_size` keys were renamed. These values are local choices, not automatic
VRAM tuning. Existing single-GPU/debug profiles remain supported.

## Resume and compatibility

Resume still requires the original `name`, output directory, compatible batch/GA,
model, data and optimizer settings. Older names such as `univtac_full` remain valid
with an explicit `name=univtac_full` override. The saved `run_id` is restored for the
W&B display name. Optional group metadata may change or be absent in older snapshots;
hardware batch defaults are compared through the resolved training batch settings.
Original snapshots and checkpoint layouts are not rewritten. Legacy non-Hydra
`experiment_name` directory behavior and deprecated `per_gpu_batch_size` are unchanged.

Only naming, configuration conversion/validation, W&B display metadata, wrapper
presets, documentation and CPU tests changed. Checkpoint loading, GPU configuration,
model/dataset/optimizer implementations and DeepSpeed initialization are untouched.

## Files changed for this task

- `configs/experiment/univtac_rgb.yaml`, `univtac_full.yaml`, `univtac_overfit5.yaml`
- `configs/hardware/dual_3090_zero3.yaml`, local `quad_3090_zero3.yaml`
- `configs/training/finetune.yaml`, `overfit_5eps.yaml`
- `gr00t/configs/hydra_config.py`, `gr00t/configs/training/training_config.py`
- `gr00t/experiment/experiment.py` (W&B arguments only), `hydra_launch.py` (resume metadata only)
- `scripts/train_univtac_sidecar.py`
- `tests/gr00t/configs/test_experiment_identity.py`, `test_hydra_config.py`
- `docs/experiment_identity.md`, link in `docs/HYDRA_CONFIGURATION.md`

Existing local A-2 loading, sensor precision/dependency, dataset statistics,
notification and model configuration edits were preserved. Existing configuration
values were retained except the requested naming/project policy, the invalid batch
key migration, hardware-derived finetune defaults and explicit overfit GA=1.
