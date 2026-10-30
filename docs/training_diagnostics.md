# Florence training diagnostics

The existing Rich dashboard and MultimodalDiagnostics callback are extended; no
loss, optimizer, clipping, precision, architecture or dataset change is made.

## Loss and action metrics

`train/loss` retains Hugging Face's normal optimization-loss meaning and rounding.
`loss/total`, `loss/action`, `loss/contact_raw`, `loss/contact_weighted`,
`loss/zero_prediction_mse`, `loss/action_to_zero_ratio`, `loss/contact_fraction`
are detached forward observations. They average microbatches over the complete
logging window and all data-parallel ranks, including accumulation. Ratios are
computed from those window means. This is the mean of each microbatch's normalized
loss, not a new pooled-valid-elements loss. No loss normalization is changed.

In normal Sidecar mode total = action + weighted contact. Warmup optimizes only
weighted contact, so `loss/action` is **zero contribution**, and
`loss/action_observed` preserves the actual flow-matching MSE for comparison;
`loss/warmup` identifies that case. Zero denominators leave ratios absent.

Action means/std are masked population statistics per microbatch (std is then
averaged, not a pooled variance). `action/valid_action_count` is mean valid scalar
entries per microbatch, `valid_dimension_count` is mean distinct valid dimensions
per example, and `valid_timestep_ratio` is the fraction of timesteps with at least
one valid dimension. `valid_action_ratio` includes both horizon and dimension
masking. The UniVTAC conversion validator specifies **8 joints** and the model's
default padded dimension is **132**: 8/132 = 0.06060606. A measured dimension count
of 8 and timestep ratio of 1 confirms pure dimension padding for a given batch;
the ratio alone cannot establish correct horizon masking or actual runtime shape.

HF/W&B retains its prefixed `train/...` diagnostic copies. Canonical `loss/*`,
`action/*`, `grad_norm/*`, `grad_status/*`, `contact/*`, `sidecar/*` are additionally
sent to the existing initialized rank-zero W&B run, using `train/global_step`.
No network logging occurs when W&B is disabled. Dashboard consumes Trainer
callbacks independently of W&B.

## Gradient timing and ZeRO-3

Installed Transformers 4.57.3 takes `train/grad_norm` from the return of gradient
clipping (ordinary training), or `engine.get_global_grad_norm()` (DeepSpeed).
Installed DeepSpeed 0.17.6 stores `_global_grad_norm = scaled_norm / loss_scale`
**before** `unscale_and_clip_grads`. Thus `grad_norm/pre_clip` is the existing
pre-clipping value, not an additional clipping operation. `clip_coefficient` is
the configured theoretical min(1, limit/(norm+epsilon)), not a measured post norm.
`clip_fraction` is the fraction of finite observed optimizer-boundary norms above
the configured limit since the last logging event; repeated epoch-end calls do
not count an update twice. `is_finite` describes the latest reported global norm.

For ZeRO-3, **post_clip is unavailable**: after step, gradients may be reused or
cleared; deriving a number from the clipping threshold would hide overflow,
rounding and optimizer-specific behavior. No extra optimizer monkeypatch is added
for post-clipping inspection. Ordinary Trainer's sampled pre-optimizer callback
already runs after clipping and can measure `post_clip` directly.

Module norms use the existing DeepSpeed optimizer-step wrapper **before** actual
step/unscale/clip. Ordinary module norms are **after Trainer clipping**. Every rank
runs the same fixed module/stage sequence. ZeRO-3 uses supported local gradient
accessors, removes partition padding using ds_numel and group rank, divides loss
scale, sums local squared norms and reduces three scalars per group. No full
module-gradient all-gather is introduced. Tiny gate gathers from the existing
callback remain. Only data parallel ZeRO-3 is supported by this interpretation;
model/tensor parallel gradient duplication would need separate validation.

Groups: dit (excluding sidecar/contact namespaces), sidecar, point_encoder,
tactile_encoder, contact_teacher, contact_predictor, contact_injection, gates,
and each sidecar_stage_N. Groups deliberately overlap; **do not sum group norms**.
`grad_status/<group>/trainable_numel`, `frozen`, `grad_present`,
`grad_buffer_present`, and `grad_finite` distinguish frozen, missing, zero and
nonfinite states. DeepSpeed may allocate zero buffers for unused parameters:
`grad_present` is unavailable there; buffer presence is not proof of autograd
participation. Frozen DiT is expected to show N/A (Frozen).

Module/contact/residual observations are sampled at the configured interval,
not necessarily every terminal logging event. The dashboard exposes module sample
step. Contact variance/norm/surprise/gate describe the **last sampled microbatch**,
mean across ranks with observations. Residual ratios average sampled microbatches
and stages across ranks. They are not presented as loss-window-weighted statistics.

## Contact teacher interpretation

Contact prediction uses Smooth L1 against **observed.detach()**, even with
freeze_contact_teacher=false. Prediction loss cannot directly update the teacher.
An unfrozen teacher/projector (and trainable tactile encoder upstream) can receive
**action-loss** gradients through observed/surprise/gating/contact injection. The
initially zero injection scales can initially block upstream action gradients.
In warmup, action is excluded, so the teacher has no direct optimization signal
from prediction loss. No freeze or weight policy is changed here.

A moving target remains possible in full mode: action-driven teacher changes can
outpace the predictor. Compare action/zero ratio, weighted contact fraction,
teacher observed norm/variance, predictor variance, group gradients, gates and
residual ratios. Existing logs alone cannot determine whether loss ~12 is action,
contact, or both. The next run's separated losses resolve that question.

## Console, files and lifecycle

The Hydra parent previously piped stdout: workers saw isatty=false even from an
SSH/Zellij terminal. For an interactive parent and dashboard not explicitly off,
workers now receive a PTY with the parent's initial window size. Rank zero owns
Rich Live; the parent relays chunks immediately (not line-buffered). Signals and
return codes retain the existing supervisor handling. This is POSIX/Linux-specific.
SIGWINCH propagates pane resizing to the worker PTY. Rich Live uses an alternate
screen with vertical cropping and a compact layout for small panes, keeping
repeated frames out of the primary terminal scrollback. Progress refreshes at
every optimizer step; metrics retain the actual logging cadence. Nonzero ranks
suppress ordinary stdout from the interactive console, preserving their full
streams in rank logs and leaving stderr/error messages available. Non-TTY keeps the existing pipe, has no forced ANSI, and
uses flushed single-line records.

In interactive launches, workers preserve original Python stdout/stderr in
`logs/rank-N-full.log` before filtering only known shard caching/waiting and
model-load sampling records from the console. Warning/error/exception/OOM/failure
messages bypass that filter. `logs/train.log` retains the relayed stream (including
Rich ANSI frames); rank logs preserve messages filtered before relay. Native code
writing directly to file descriptors bypasses the Python filter and remains in
train.log. Debug sampling env flags are not disabled. These logs may be large.

Rich uses normal refresh, with progress, speed, elapsed/ETA, loss, gradient, action,
optimizer and status panels. Saving is marked at the scheduled save boundary;
on_save restores Running and records the completed checkpoint. Existing Trainer
finally cleanup restores terminal and handlers on exception/KeyboardInterrupt;
normal completion and resume initialize fresh display state. A SIGKILL cannot
execute cleanup. Loss stays visible while save blocks; it is not a byte-level
checkpoint-progress meter.

## Configuration and next run

Overfit now enables basic module diagnostics every 50 steps; full-data defaults
remain off. Existing validation/schema fields are reused, with no new Hydra key.
Resume permits changes to diagnostic enable/interval/level metadata only; semantic
batch, architecture and optimizer compatibility checks remain unchanged.

```bash
bash scripts/train_univtac_sidecar.sh overfit5 \
  hardware=quad_3090_zero3 \
  training.enable_training_dashboard=true \
  training.enable_multimodal_diagnostics=true \
  training.multimodal_diagnostics_interval=50 \
  training.multimodal_diagnostics_level=basic
```

Use `dry_run=true` first to inspect settings. Do not change batch/GA/LR or teacher
settings to interpret this instrumentation comparison. For a resume, retain the
original name/output and normal resume flags.

CPU tests cover decomposition/warmup, 8/132 and horizon masks, detached update
invariance, accumulation window reduction, canonical mocked W&B logging, gradient
finite/missing/zero states, scaled/padded ZeRO partition scalar reduction, callback
cleanup, rank-zero display, PTY detection, non-TTY and checkpoint status. A tiny
2-step CPU Trainer compares diagnostics on/off parameter updates exactly. Actual
4-GPU ZeRO-3 updates, watchdog safety, terminal rendering in the user's SSH/Zellij,
performance overhead and the cause of the observed loss spike remain unverified.

## Validation and changed files

Focused CPU suite: **114 passed** (96 dependency/configuration warnings):
`test_multimodal_instrumentation.py`, `test_sidecar_dit.py`, `test_hydra_config.py`,
`test_experiment_identity.py`, `test_resume_compatibility.py`. The overfit wrapper
also composed successfully with quad hardware and `dry_run=true`. Ruff and
`git diff --check` passed. No GPU training, checkpoint downloads or pushes ran.

Changed files:

- `gr00t/experiment/action_diagnostics.py` (detached statistics and ratios)
- `gr00t/experiment/multimodal_diagnostics.py` (existing sampled gradient observer)
- `gr00t/experiment/trainer.py` (window reduction, clipping metadata, W&B aliases)
- `gr00t/experiment/training_dashboard.py` (existing Rich display)
- `gr00t/experiment/console_logging.py` (filtered console, complete worker stream)
- `gr00t/experiment/hydra_launch.py` (PTY relay and diagnostic-only resume metadata)
- `scripts/train_hydra_worker.py` (interactive stream setup)
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` (forward diagnostics only)
- `gr00t/model/extension/sidecar_dit.py` (detached teacher norm and stale-record reset)
- `configs/training/overfit_5eps.yaml` (enable existing diagnostics option)
- `tests/gr00t/experiment/test_multimodal_instrumentation.py`
- `docs/training_diagnostics.md`
