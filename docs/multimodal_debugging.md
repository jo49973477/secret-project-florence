# Multimodal training instrumentation

These tools observe training; they do not change gates, learning rates, losses,
optimizer groups, checkpoint formats, or the model's mathematical forward pass.

## Terminal dashboard

The default (`enable_training_dashboard=None`) uses one Rich Live display when
stdout is a TTY. It shows training progress, elapsed time, ETA, step/s, the latest
training measurements, and evaluation progress/metrics. Missing values show
`N/A`. Only global rank zero renders. Rich coordinates stdout/stderr writes and
handles resizing; normal logging, warnings, notifications and tracebacks remain
available. Trainer history and reporting callbacks (including W&B) still receive
`Trainer.log()` events.

Both HF `ProgressCallback.on_log()` and `PrinterCallback.on_log()` print metric
dictionaries. The trainer replaces those renderers rather than suppressing
logging globally. Explicitly disabling the dashboard on a TTY retains a tqdm bar
with its dictionary-printing method disabled. Non-TTY runs, including `tee`,
always print clean metric lines without a progress bar or escape sequences.

Add these implemented options to an existing `launch_finetune.py` command:

```bash
# Interactive when stdout is a TTY; the bool-or-None option requires a value.
--enable-training-dashboard True --dashboard-refresh-seconds 1.0

# Disable the metrics panel (retain a quiet tqdm bar on a TTY).
--enable-training-dashboard False
```

Omit the option for automatic detection. Rendering uses already logged CPU
scalars and is throttled by `dashboard_refresh_seconds`; refreshing does not
read model parameters. Training/evaluation exceptions and Ctrl-C close the
display before propagating. `SIGKILL` cannot run Python cleanup.

## Basic and per-layer diagnostics

Disabled by default. Add the following to an existing fine-tune invocation:

```bash
--enable-multimodal-diagnostics \
--multimodal-diagnostics-interval 50 \
--multimodal-diagnostics-level basic
```

Use `--multimodal-diagnostics-level per_layer` for complete layerwise gate,
gate-gradient, optimizer-update delta, trainability, finite-value, and residual
ratio records. The terminal always shows compact aggregates. Disable with
`--no-enable-multimodal-diagnostics` or `--multimodal-diagnostics-level off`.
`off` takes precedence over the enable flag.

Both existing UniVTAC launch scripts now forward additional CLI arguments:

```bash
# Set these paths to your existing data and model snapshot.
export DATASET_PATH=/path/to/univtac_lerobot
export BASE_MODEL=/path/to/GR00T-N1.7-3B

EXPERIMENT=C uv run bash scripts/launch_univtac_controlled_experiment.sh \
  --enable-multimodal-diagnostics \
  --multimodal-diagnostics-interval 50 \
  --multimodal-diagnostics-level basic

EXPERIMENT=C uv run bash scripts/launch_univtac_controlled_experiment.sh \
  --enable-multimodal-diagnostics \
  --multimodal-diagnostics-interval 50 \
  --multimodal-diagnostics-level per_layer

EXPERIMENT=C uv run bash scripts/launch_univtac_controlled_experiment.sh \
  --no-enable-multimodal-diagnostics --enable-training-dashboard False
```

The scripts pipe output through `tee`, so they use clean text, even if the
interactive option is True. For an interactive dashboard run the underlying
`uv run python gr00t/experiment/launch_finetune.py` or `uv run torchrun ...`
command directly in a terminal without that pipe. The full multimodal launcher
also accepts these additional options:

```bash
uv run bash scripts/finetune_univtac_full_multimodal.sh \
  --enable-multimodal-diagnostics --multimodal-diagnostics-level per_layer
```

Records are appended once per sampled optimizer step, independently of
`logging_steps`, to `<output_dir>/multimodal_diagnostics.jsonl`. Numeric
aggregates go through the existing Trainer logging/reporting infrastructure;
there is no separate W&B client. `layers` contains stable zero-based indices.
Existing logs are appended on resume. Sample-to-sample deltas start unavailable
after restarting the process; individual optimizer-update deltas remain valid.

### Meaning and timing

* Gate mean/max absolute values, fraction within `1e-8` of zero, trainability,
  optimizer membership, gradient norms and finite flags are measured separately
  for point and tactile gates.
* Gate values and gradients are sampled **before** the optimizer update.
  `*_gate_update_norm` and layerwise `*_gate_delta` measure the actual sampled
  update; `*_gate_sample_delta_norm` measures change between sampled post-update
  values. The first sample has no preceding sampled value.
* Encoder parameter gradient norms cover the entire shared encoder once.
  Cross-attention norms cover each modality's shared attention module once.
  They are not repeated as fictitious per-layer encoder norms.
* `*_residual_ratio_mean/max` summarize the norm of each actual gated residual
  divided by the entering action hidden-state norm (FP32, denominator floor
  `1e-12`). Layer ratios average observed microbatches and ranks; aggregates are
  the mean/max of those layer means. They are detached scalars from the actual
  forward, not a second attention evaluation. Checkpoint recomputation is not
  counted a second time.
* Ordinary Trainer sampling occurs after backward, unscale and clipping, before
  optimizer step/zero_grad. DeepSpeed sampling brackets the initialized ZeRO
  optimizer's real `step()` inside Accelerate backward, before DeepSpeed's own
  unscale/clipping. Its reported gradients are divided by the active loss scale.
  `gradient_timing` and `gradient_scale` document this difference.

A zero gate can have a nonzero gradient while encoder/attention gradients are
zero. After gates become nonzero, those branches can receive gradients. A single
initial zero observation does not establish starvation; compare records across
steps and check trainability, optimizer membership and actual gate deltas.
Large residual ratios mean a strong injection relative to the entering hidden
state, not necessarily an error. No heuristic automatically changes training.
Non-finite observations produce warnings and JSON `null`; ordinary missing
measurements are unavailable, not numerical zero.

### ZeRO-3 and overhead

Every rank participates in safe gate access and collectives. Only tiny gate
vectors use `GatheredParameters` for full-value access, reading the actual model
parameter rather than an FP32 optimizer master. Encoder/attention norms stream supported local
ZeRO-3 gradient shards, exclude partition padding, and reduce scalar sums.
Already-global gradients are not summed again. Optimizer membership uses the
original low-precision parameter groups, not flattened FP32 master identities.

DeepSpeed may allocate gradient buffers for unused parameters. For ZeRO-3,
`grad_present` is therefore `null`, `grad_buffer_present` describes availability,
and `grad_status` distinguishes `zero_buffer`, `nonzero_buffer`, `missing` and
`invalid`. An informational timing message explains this limitation. A buffer
norm alone cannot establish autograd participation. Unsupported/unavailable gate
values or gradient norms are warned about rather than invented.

Reading a partitioned parameter directly can yield empty/local representations;
that is unsuitable for gate statistics. The instrumentation avoids this read.
The historical `60397977600.0` anomaly has **not** been reproduced or explained.

When disabled, no activation/gradient hooks are registered and no extra norms,
collectives, parameter snapshots, or JSONL writes run. Enabled sampling adds
FP32 residual reductions, gradient reads, scalar communication and CPU/JSON
work at the requested interval; large encoders and optimizer offload can make
samples noticeable. Only detached residual scalars and small gate snapshots are
retained. Exact layerwise encoder-gradient attribution and activation-gradient
hooks are intentionally omitted: they would need more retained state or extra
backward work. No universal runtime percentage is claimed.

## Standalone pretrained DiT verification

Use a **local HF snapshot** with `config.json` and safetensors or PyTorch weights
(single file or sharded index). HF cache snapshot symlinks are supported. The
script does not download models or start training.

```bash
# All real original DiT tensors, Action DiT only; no VLM/encoder allocation.
uv run python scripts/verify_pretrained_dit.py \
  --reference-checkpoint /path/to/GR00T-N1.7-3B \
  --dtype bfloat16 --report outputs/dit_weights.json

# Add three deterministic zero-gate parity cases on the real Action DiT.
uv run python scripts/verify_pretrained_dit.py \
  --reference-checkpoint /path/to/GR00T-N1.7-3B \
  --dtype bfloat16 --forward-parity --device cuda:0 \
  --report outputs/dit_integrity.json

# Audit the actual full training model-loading path using a saved repository
# Config YAML (model/training/data sections, not the flat FinetuneConfig CLI).
# This calls Gr00tN1d7Pipeline._create_model(), not dataset/training setup.
uv run python scripts/verify_pretrained_dit.py \
  --reference-checkpoint /path/to/GR00T-N1.7-3B \
  --training-config /path/to/config.yaml \
  --forward-parity --device cpu --report outputs/dit_pipeline.json

# Optional saved-file comparison; use a candidate saved BEFORE training.
uv run python scripts/verify_pretrained_dit.py \
  --reference-checkpoint /path/to/GR00T-N1.7-3B \
  --candidate-checkpoint /path/to/initialized-multimodal \
  --report outputs/dit_saved_weights.json
```

The default audit constructs the real Action DiT using the checkpoint's resolved
configuration and loads its real tensors with PyTorch. It shares key validation
and adapter classification with the training setup. `--training-config` instead
uses the repository's actual HF AutoModel loading path, loading-info validation,
and separate sensor validation/bootstrap, and reports its effective checkpoint,
architecture and `skip_weight_loading`. It can require substantial CPU RAM and
external sensor checkpoints; it does not substitute a tiny model. Saved-file
comparison cannot establish how training initialized a model, so loader status
is explicitly SKIP and skip-weight-loading is unknown.

The expected set comes from the original architecture, including every original
Transformer block, timestep encoder, normalization and output projection. Every
expected tensor is compared, with per-tensor shape, dtype, finite flag, absolute
and safe relative error, equality, per-block status, coverage and worst ten
errors in JSON. Missing reference keys cannot disappear from the expected set.
Only exact new adapter prefixes are exempt; sensor encoders have separate
loading reports.

The reference is cast to the candidate dtype **before** comparison. Default
weight tolerances are zero, because loading should perform only the documented
dtype cast; a small modified original parameter must still fail. Explicit
`--atol`/`--rtol` overrides are recorded. Relative errors use denominator
`max(abs(cast_reference), candidate_dtype_epsilon)`.

Forward parity uses eval/inference mode, fixed inputs/masks and timesteps
0/127/999, with gates set exactly zero. It compares final outputs and hidden
states using `atol=rtol=1e-5` for FP32 and `2e-2` for BF16/FP16 (allowing rounding
at lower precision). The reference runs first and is offloaded before moving
the candidate to GPU. It tests the DiT, not the final action decoder. No second
full 3B VLM is allocated. Weight reads are lazy per safetensor or mmap-backed
PyTorch shard. Instantiated DiTs still need host RAM; forward time depends on
the real architecture, device and dtype and may take minutes on CPU.

Statuses and exit codes:

* `PASS`: a check completed and passed. Overall PASS requires weight, loading,
  and forward checks to pass (exit 0).
* `FAIL`: missing/changed/non-finite/incompatible original weights, invalid
  loading, or failed numerical parity (exit 1). Errors retain tracebacks.
* `SKIP`: a check was not requested or could not run, with a precise reason.
  Disabled encoders are reported separately and are not required for DiT scope.
  Skipped required checks make overall `INCOMPLETE` (exit 2), even if all
  completed checks passed. CUDA unavailability/OOM never becomes a parity PASS.

The ordinary tests use explicitly small fixtures. An optional real-checkpoint
test is available without downloads:

```bash
GR00T_DIT_INTEGRATION_CHECKPOINT=/path/to/GR00T-N1.7-3B \
  uv run python -m pytest tests/gr00t/model/test_dit_integrity.py \
  -k optional_real_checkpoint -v
```

A passing fixture test is not evidence that the real GR00T checkpoint passes.

## Validation recorded for this implementation

The final focused suite passed **22 tests**, with the optional real-checkpoint
integration test skipped. The two-step HF Trainer smoke test uses gradient
accumulation, records one sampled optimizer step, and checks exact equality of
all model weights with diagnostics enabled versus disabled. Tests also cover
zero-gate parity, changing conditioning with nonzero gates, missing/changed
weights, dtype casting, gate/encoder gradients, local-shard reductions through
mocks, JSONL, TTY/non-TTY rendering, evaluation display and interrupt cleanup.

The broader model/trainer run had **100 passes, one skip and one failure** in
`TestActionHeadTrainableParams.test_concerto_backbone_and_projection_have_independent_freeze_controls`:
its existing fake Concerto encoder has `architecture_config=None`, while the
existing action-head constructor calls `dict(architecture_config)`. A separate
configuration regression run failed
`test_param_offload_and_subgroup_are_independent_opt_ins`: it expects no
`sub_group_size`, but the existing working-tree ZeRO-3 config sets `50000000`.
These pre-existing fixtures/training settings were left unchanged. Manifest
alignment also reports a pre-existing `h5py` presence mismatch across platforms;
Rich itself is aligned. Scoped Ruff checks/formatting, shell syntax, CLI help,
and `uv lock --locked` passed.

No full training experiment, real GR00T checkpoint verification, real multi-GPU
ZeRO-3/offload execution, live W&B/notification delivery, or manual SSH/terminal
resize test was run. The hardware paths use supported safe access APIs and
local/mock tests, not a claim of validated hardware integration.
