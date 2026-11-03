# Diagnostics

Run these from the repository root when investigating a specific issue. They
are not normal training entrypoints.

- `benchmark_checkpoint_io.py`, `summarize_model_load.py`, and
  `smoke_distributed_model_loading.py`: checkpoint and distributed loading.
- `diagnose_concerto_precision.py`, `diagnose_fixed_batch.py`,
  `smoke_test_concerto_point_encoder.py`, and `smoke_test_sparsh_tactile.py`:
  sensor and batch diagnostics.
- `validate_hf_config_alignment.py` and `verify_droid_rotation_correction.py`:
  configuration and data checks.

Example: `uv run python scripts/diagnostics/validate_hf_config_alignment.py`.
Some checks require local model files or GPUs; inspect their help before use.
