# Hydra migration implementation report

The migration adds a Hydra parent launcher and a direct, typed conversion into existing
GR00T Config objects. The existing experiment.run API, model setup loader, Trainer
optimizer ownership and checkpoint state restoration are retained. Pre-existing local
changes were preserved. No training job, dataset conversion or real subset preparation
was executed during implementation.

## Added configuration tree

```text
configs/
├── model/{gr00t_base,gr00t_multimodal}.yaml
├── data/{univtac_rgb,univtac_multimodal,so100}.yaml
├── training/{finetune,overfit_5eps,debug}.yaml
├── hardware/{single_3090,dual_3090_zero3}.yaml
└── experiment/{univtac_full,univtac_rgb,univtac_overfit5,so100_baseline}.yaml
```

## Files in this change

Added:

- The 14 YAML files above.
- gr00t/configs/hydra_config.py: supported-field mappings, OmegaConf type validation,
  modality/enumeration construction, compatibility checks and checkpoint assertions.
- gr00t/experiment/hydra_launch.py: preflight, dataset prerequisites, checkpoint validation,
  strict resume comparison, immutable parent snapshots, metadata and process supervision.
- gr00t/experiment/optimizer_artifacts.py: constructed group counts and post-restoration LRs.
- scripts/train_hydra.py and scripts/train_hydra_worker.py: one Hydra parent, shared snapshot
  loading in workers, checkpoint metadata inheritance and existing experiment.run invocation.
- scripts/univtac_overfit_workflow.py: explicit, nondestructive five-episode preparation and
  same-episode evaluation commands; native ZeRO export remains an explicit prerequisite.
- tests/gr00t/configs/test_hydra_config.py and tests/gr00t/experiment/test_hydra_artifacts.py.
- docs/HYDRA_CONFIGURATION_INVENTORY.md, docs/HYDRA_CONFIGURATION.md and this report.

Modified by this migration:

- pyproject.toml, scripts/deployment/{jetson,spark}/pyproject.toml and uv.lock: hydra-core 1.3.2.
  Orin/Thor manifests are existing symlinks to Jetson's manifest.
- gr00t/experiment/experiment.py: resolved Hydra config in W&B; optimizer artifact callback.
- gr00t/experiment/trainer.py: retain a machine-readable description of existing optimizer
  groups. Group assignment logic and optimizer calculations are unchanged.

Other modifications visible in git status predate this work. In particular, model files,
legacy shell scripts, legacy config defaults and DeepSpeed JSON were not edited by this migration.

## Precedence

Hydra group defaults → experiment overrides → CLI overrides → strict supported-field and
type validation → direct GR00T dataclass conversion. Model tuning and processor/data settings
are intentional overrides along the existing setup.py path. Checkpoint architecture is
inherited; checkpoint_expectations asserts compatibility and cannot resize weights.
The worker checks embedded sensor and LoRA architecture before model loading. setup.py keeps
its strict tensor checks, embedded sensor validation and deferred bootstrap logic.
Full training resume separately restores model/ZeRO, optimizer, scheduler, RNG and counters;
requested LR changes are rejected, and the post-restore current group LRs are reported.

## Commands

```bash
FLORENCE_DATASET=/path/to/converted_univtac CUDA_VISIBLE_DEVICES=0,1 \
uv run python scripts/train_hydra.py --config-name experiment/univtac_full

uv run python scripts/train_hydra.py --config-name experiment/univtac_full \
  training.max_steps=2000 training.action_head_learning_rate=5e-5 dry_run=true

uv run python scripts/univtac_overfit_workflow.py prepare \
  --source /path/to/univtac --target /path/to/new_overfit5

FLORENCE_OVERFIT_DATASET=/path/to/new_overfit5 CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/train_hydra.py --config-name experiment/univtac_overfit5 \
  hardware=single_3090 output_dir=outputs/memorization

uv run python scripts/univtac_overfit_workflow.py evaluate \
  --dataset /path/to/new_overfit5 --output outputs/memorization --gpu 0
```

The [configuration guide](HYDRA_CONFIGURATION.md) contains override, encoder, stage2/stage3,
resume, resolved-config inspection and saved-snapshot reproduction commands.

## Executed checks

- New CPU suites: **51 passed**, with no pretrained downloads or CUDA initialization.
  Includes subprocess guards against importing training or calling AutoModel/AutoConfig/
  AutoProcessor.from_pretrained, CUDA init and distributed init during dry-run.
- Focused new + existing regressions: **115 passed, 1 deselected**. Covers optimizer ownership
  and LR scheduling/state loading, resume strictness, warnings, safe YAML and DeepSpeed config.
- A broader earlier run found two legacy failures, both reproduced in isolation without
  invoking Hydra: test_univtac_rgb_and_multimodal_share_common_modalities registers
  new_embodiment twice; test_param_offload_and_subgroup_are_independent_opt_ins conflicts
  with the pre-existing local zero3_config.json sub_group_size=50000000. The latter was
  deselected in the passing focused run. The former legacy module was tested separately;
  the new modality parity regression compares the shared builder without mutating registration.
- Ruff lint and format checks: pass for all ten Python files added/modified here.
- git diff --check: pass.
- uv lock --locked --offline: pass with writable /tmp cache.
- CLI smoke checks: resolved config inspection, LR/step overrides, single-GPU group switching,
  SO100 preflight and preflight reproduction from a saved snapshot all pass.
- pre-commit run --all-files: could not run; pre-commit is not installed. Equivalent targeted
  Ruff checks ran. Manifest alignment reports only pre-existing h5py presence drift; the
  new hydra-core dependency is aligned across every platform manifest.

## Deliberate differences and remaining limits

- Multimodal YAML uses dataset horizon 40 as requested; the current Python builder uses 16.
  Set data.modalities.action.horizon=16 for equivalence. Model maximum remains 40 and is
  verified against checkpoint metadata in workers.
- XYZRGB dimension is six. The inspected local full script already uses six; its internal
  model dataclass still defaults to three. Hydra supplies six explicitly and rejects mismatches.
- RGB recipes explicitly disable inactive conditioning flags; the legacy alternate DiT ignored
  their True defaults. Normalization, tuning and action representations otherwise match.
- Portable dataset placeholders replace machine-specific defaults. Subset preparation refuses
  an existing target, rather than rebuilding/deleting it. SO100 baseline retains the common
  five checkpoints; its legacy overfit shell keeps all saves.
- Full Hydra resumes require Hydra provenance. Legacy runs retain their legacy resume launcher;
  model-only/standalone-loadable weights can seed a fresh Hydra fine-tune. No silent state reset.
- Inactive legacy controls are rejected. The current DatasetFactory cannot supply training-time
  evaluation splits; separate open-loop evaluation is documented and retained.
- Native ZeRO checkpoints need explicit inference export before same-episode evaluation. The
  wrapper fails with an actionable instruction instead of treating partitions as HF weights.
- Existing ignore_data_skip=True and sampler behavior remain unchanged; full resume is not a
  guarantee of exact sample-order replay. Extending max_steps uses the existing HF scheduler
  reconstruction/restoration policy, which may differ from a run planned for that length.
- Checkpoint preflight checks state families and step metadata, not tensor contents or complete
  partition integrity. Actual restoration is still owned by Trainer/DeepSpeed.
- No GPU/real Torchrun training, multi-rank interruption, ZeRO2/ZeRO3 resume, external Concerto/
  Sparsh downloads, model memory-fit or convergence tests were executed. CPU tests exercise
  command construction and process behavior with mocks. Do not infer distributed verification.
- Reproducibility records paths, commit/dirty flag and config, not hashes/backups of datasets,
  remote weights or local changes. Keep those assets/revisions available.

Existing dataclass defaults remain authoritative for unexposed internal settings; checkpoint
architecture remains checkpoint-owned. DatasetFactory's fixed single-turn/no-eval behavior,
Trainer's data-skip policy, official sensor bootstrap mechanics and default DeepSpeed JSON
remain in their existing implementations to avoid changing training behavior in a config
refactor. No model equations, neural architecture implementation, losses or optimizer
ownership rules were modified by this work.
