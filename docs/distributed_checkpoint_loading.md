# 4-rank checkpoint loading: I/O, memory, and synchronization

This change targets host loading before the existing DeepSpeed setup. It does
not move ZeRO-3 initialization, change the NCCL training group, model/Sidecar,
DiT weights, BF16 policy, loss, optimizer, learning rates, datasets, or GPU count.
Existing working-tree changes, deferred sensors, strict loading checks, and the
Concerto sparse FP16 operation boundary are preserved. Actual host four-GPU
loading and source-checkpoint DiT integrity were validated; engine/step testing
is explicitly separate below.

## Actual initialization order

1. `train_univtac_sidecar.py` delegates to `train_hydra.py`. The parent validates
   sensor imports and starts Torchrun. Workers inherit checkpoint architecture
   metadata without loading its weights.
2. `experiment.run` initializes its **existing NCCL process group**, sets the
   seed, and calls `pipeline.setup()`.
3. `Gr00tN1d7Pipeline._create_model()` reads GR00T config and sensor layout.
   Existing coverage validation checks actual Safetensors headers before
   skipping the redundant external Cosmos VLM weight load. Incomplete coverage
   still falls back; strict missing/unexpected/mismatched validation is retained.
4. Model loading is serialized rank 0, 1, 2, 3. Transformers 4.57.3 already uses
   `init_empty_weights` and Safetensors mapping in its ordinary low-memory path.
   `low_cpu_mem_usage=True` is already set. Its meta/header inspection is distinct
   from each shard's later parameter materialization; repeated header opens do
   not imply repeated full-shard tensor reads.
5. Existing explicit loading validation runs. Only validated missing Sidecar
   tensors use their original adapter constructors. RoPE and patch-memory-format
   restoration remain in place. No DiT tensors are rewritten by this step.
6. Embedded sensors are validated without external bootstrap. Newly enabled
   sensors bootstrap **after** GR00T loading, still one rank at a time. Concerto
   reads its official config/state dict, constructs its original architecture,
   and assigns weights. SPARSH uses the local ViT and strict original checkpoint
   keys. Each rank retains its own model; repeated rank reads are currently
   required, not an accidentally repeated load in one rank.
7. The pipeline constructs processor/dataset, and returns to `experiment.run`.
   **Only afterwards** does the existing code obtain DeepSpeed config and create
   `TrainingArguments`, Trainer, and the ZeRO-3 engine. An initialized NCCL
   process group during loading does not mean ZeRO-3 is already partitioning it.

If a caller unexpectedly activates Transformers' ZeRO-3 integration before the
serialized loader, the pipeline now fails clearly: that loader's internal
collectives require every rank, which is incompatible with a one-rank callable.
It does not move initialization or silently turn on simultaneous disk reads.

## Confirmed resource constraint and BF16 storage correction

Host discovery outside the sandbox found eight RTX 3090 devices; the smoke test
used only free devices 0–3. Host available RAM was 194,186 MiB (about 189.6 GiB), but
`/sys/fs/cgroup/user.slice/user-1020.slice/memory.max` was **34,359,738,368 bytes
(32 GiB)**, shared with the user's other processes. Session/user cgroup OOM-kill
counters were present. Host free RAM is therefore not the job's memory budget.

The first real loading attempt retained original BF16 action-head checkpoint
values in FP32 CPU Parameters on each rank. Rank 2 stalled inside shard 2;
rank 0 was SIGKILLed (-9), with I/O PSI full avg10 observed at 34.10% and
avg60 at 51.93%. Cgroup memory-limit/reclaim pressure and OOM counters are
consistent with the termination; the exact PID's kernel OOM log was not
accessible. This was **not an observed NCCL timeout** in the new Gloo path.

The subsequent fix preserves original checkpoint dtype without changing RNG:

- Header inspection also collects actual **BF16** `action_head.*` keys/shapes.
- All original constructors and initialization calls still execute normally.
- At the end of model construction, **before** HF weight materialization and
  before ZeRO-3, only matching **meta Parameters** are prepared as BF16 using
  normal module conversion machinery. No real CPU Parameter data is modified.
- Source FP32 tensors, shape mismatches, buffers, the VLM's explicit FP32 tuning
  settings, and missing/new Sidecar Parameters remain untouched. An active
  Transformers ZeRO-3 integration disables this preparation.
- HF then assigns original BF16 checkpoint storages directly instead of creating
  redundant per-rank FP32 CPU copies. New Sidecar initialization stays FP32 until
  the existing training setup converts it as before. CPU RNG and new adapter
  values are identical in regression tests.

This selected **537** Parameters per rank in the real checkpoint. All **456
original DiT tensors** matched source values exactly on **all four ranks**;
maximum absolute and relative errors were both zero. Mixed FP32/BF16 checkpoint
and no-buffer/no-live-Parameter/active-ZeRO modification regressions are covered.
Hardware BF16/FP16 settings, later engine conversion, loss, and optimizer are
unchanged. `GROOT_MODEL_LOAD_PRESERVE_BF16=0` disables just this new preparation;
**do not repeat that comparison under the same constrained cgroup** without a
separate memory plan. The original failing run's partial records were retained
rather than deliberately causing another OOM.

## Eliminated copies and intentionally retained work

The external VLM duplicate-load bypass was already implemented and is retained;
this change does not claim to have newly removed that whole checkpoint load.
SPARSH Safetensors already use `safetensors.torch.load_file`; that existing mapping
is retained, without an additional streaming implementation.

The newly optimized copies were eager `torch.load` tensor-storage reads:

- Deferred Concerto bootstrap now resolves the **same** official `.pth` path
  and reads it with PyTorch's existing `mmap=True`. This replaces the upstream
  eager `concerto.load(..., ckpt_only=True)` read rather than calling both.
  `weights_only=True`, config overrides, strict key/shape checks, and architecture
  construction/assignment behavior are preserved. Standalone and compatibility
  paths retain the upstream reader unless explicitly opted into mapped loading.
- SPARSH `.pt`/`.pth`/`.ckpt` bootstrap uses the same mapping when the file is
  ZIP-serialized. Its previous `weights_only=False` policy is preserved; there
  is no unsafe-unpickle retry or fallback after an error. Legacy non-ZIP formats
  continue through the original eager format reader. Safetensors are unchanged.

`model.optimize_pretrained_loading=true` also selects mapped deferred sensor
reads. `GROOT_MODEL_LOAD_MMAP=0` disables just the new torch-storage mapping for
controlled comparisons while keeping the existing VLM optimization enabled.
Tests verify exact loaded values, trainability, and unchanged CPU RNG state,
including newly initialized projection values. The BF16 preparation above is
also controlled by `model.optimize_pretrained_loading` and only selected when
training BF16 is enabled; source FP32 checkpoint values are never downcast.

No full state-dict clone or tensor broadcast is added. Renaming/filtering a
state-dict via a Python dictionary creates references, not copies of its tensor
storage. Existing `assign=True` already avoids normal sensor checkpoint-to-model
copies; the special Concerto outer-meta-context `to_empty`/copy fallback and its
non-persistent-buffer repair are retained because their tests show they are
necessary. Generic removal of those copies would break checkpoint integrity.

Sensor constructors still temporarily allocate their normal FP32 parameters
before strict assignment. They also advance the RNG before projection/adapter
initialization. This change deliberately does not replace them with a blanket
meta constructor: exact RNG behavior and non-persistent buffers would need a
separate validated solution. Likewise no global BF16 construction is introduced
for the pretrained DiT. This limits the RAM saving; it is not hidden as a solved
part of the optimization.

## Why the old wait could time out NCCL

Previously every inactive rank entered a default-group NCCL status `all_reduce`
while the active rank was still inside disk/model loading. A long rank-2 read
stall can therefore leave a queued NCCL operation pending on ranks 0, 1, and 3
until the watchdog expires. The observed SATA I/O pressure, I/O wait, and
`folio_wait_bit_common` are consistent with this path. They alone do not prove
that this was the only cause of the reported timeout; actual rank traces are
still required.

The loading helper retains **all serialization and error synchronization**:
active rank body → status reduction → error broadcast if needed → barrier →
next rank. With an NCCL default group, those small control operations now use
one lazily created, cached **CPU/Gloo side group**. No weights are sent through
it. CPU waits no longer enqueue a load-completion collective on NCCL. The
training communicator and its initialization point are unchanged. Group creation
is collective and must occur in the same order on every rank.

`GROOT_SERIALIZED_LOAD_CONTROL_BACKEND=default` restores the previous control
backend for diagnosis. `gloo` is the new default. An existing Gloo process group
is reused directly. No global NCCL timeout is raised, and no synchronization is
removed. Gloo still has a finite timeout: a permanently stuck or killed rank,
filesystem failure, unavailable Gloo, or mismatched rank control flow remains an
error. This change does not repair the SSD or a hung process. Existing non-loader
`run_or_wait_on_rank0` operations keep their default-group behavior and now have
optional entry/exit traces too.

## Rank-0 single-read distribution: analyzed, not implemented

A full-state Python object broadcast would create large serialization buffers,
potentially retaining the checkpoint plus another full copy on rank 0 and copies
on receivers. With the reported 4.99 + 1.92 GB shards, that can add multi-GB
transients. NCCL cannot broadcast CPU tensor storages directly, so a custom
NCCL solution also needs bounded CUDA staging and carefully matched dtype/key/
collective ordering. A Gloo tensor-wise scheme would avoid GPU staging, but all
ranks still require complete CPU models before the **unchanged** ZeRO-3 setup.
It would not remove those resident model copies.

Transformers' ZeRO-3-aware loader does have a separate collective path, but
activating it earlier would change the prohibited initialization timing. Custom
rank-0 tensor streaming requires independent four-GPU integrity, buffer-lifetime,
and peak-memory validation. It is therefore **not implemented**. File-backed
mapping can share clean kernel page-cache pages on this single host without a
new distribution protocol; it does not guarantee less physical disk traffic.
No original Hugging Face cache file is rewritten, resharded, or deleted.

## Opt-in tracing and comparison

```bash
export GROOT_DEBUG_MODEL_LOAD_IO=1
export GROOT_DEBUG_MODEL_LOAD_MEMORY=1
export GROOT_MODEL_LOAD_TRACE_DIR=/tmp/gr00t-load-candidate
# Use your existing Sidecar command with hardware=quad_3090_zero3.
```

Each JSONL record includes UTC and monotonic time, global rank, PID, RSS and
VmHWM peak, `/proc/self/io` `read_bytes`/`rchar`, stage, and entry/exit. Loader
records include resolved checkpoint filename and file size. Exit records include
elapsed time and I/O deltas, including exception paths. There are timings for
actual Transformers shard materialization, low-level reads, full GR00T loading,
sensor bootstrap, Sidecar initialization, finalization, control collectives,
and a marker immediately before existing TrainingArguments/ZeRO-3 setup.

The dedicated logger preserves opted-in events even though nonzero ranks set
the root logger to WARNING. Optional files are per rank/PID. Use `/tmp` or another
logging volume to avoid adding debug writes to the checkpoint SSD. The diagnostic
scope temporarily wraps existing loader functions and restores them in `finally`;
it neither replaces loading implementations nor retains state dictionaries.
It adds timing/proc/log overhead only when enabled. Physical I/O from mapped pages
occurs when touched; `transformers.load_shard_file`, not just its header-loading
subcall, is the relevant end-to-end interval.

For an isolated sensor-I/O baseline use a separate output/trace directory and
keep the same Hydra config, seed, checkpoint, and hardware:

```bash
GROOT_MODEL_LOAD_MMAP=0 GROOT_MODEL_LOAD_PRESERVE_BF16=1 GROOT_SERIALIZED_LOAD_CONTROL_BACKEND=gloo
```

Candidate settings are `GROOT_MODEL_LOAD_MMAP=1`,
`GROOT_MODEL_LOAD_PRESERVE_BF16=1`, and
`GROOT_SERIALIZED_LOAD_CONTROL_BACKEND=gloo`. This isolates mmap rather than
reenabling the FP32 memory problem. The old NCCL control backend is separately
available via `GROOT_SERIALIZED_LOAD_CONTROL_BACKEND=default`; it was not used
to deliberately reproduce a watchdog timeout. Do not set
`model.optimize_pretrained_loading=false` for an isolated new-I/O comparison:
that also reenables the earlier external VLM load. Compare traces with:

```bash
.venv/bin/python scripts/summarize_model_load.py /tmp/gr00t-load-candidate \
  --baseline /tmp/gr00t-load-baseline
```

The report separates nested phases rather than adding their overlapping I/O
counters together. `read_bytes` measures physical block reads and may be zero
on a warm cache; `rchar` is userspace read traffic and excludes mmap page faults.
VmHWM is process-lifetime peak. Summing rank peaks is neither simultaneous host
RAM nor physical memory usage, because mapped/file-cache pages can be shared.

## Validation and measured limits

Focused CPU commands:

```bash
NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. .venv/bin/python -m pytest \
  tests/gr00t/utils/test_checkpoint_io.py \
  tests/gr00t/model/test_pretrained_loading.py \
  tests/gr00t/model/test_concerto_checkpoint_loading.py \
  tests/gr00t/model/test_sparsh_tactile_encoder.py \
  tests/gr00t/model/test_dit_integrity.py -m 'not gpu' -q

NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. .venv/bin/python -m pytest \
  tests/gr00t/utils/test_serialized_loading.py \
  tests/gr00t/utils/test_run_or_wait_on_rank0.py -q

NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. .venv/bin/python \
  scripts/benchmark_checkpoint_io.py --size-mib 32
```

The four-rank test uses **CPU/Gloo**, real serialized tiny checkpoint reads,
rank-2 delay/failure propagation, unchanged RNG during group creation, and
entry/exit traces. It mocks backend selection to exercise creation of the Gloo
side group; it does not run NCCL or simulate CUDA kernels. The sandbox blocked
loopback socket creation; the permitted local-socket test run passed. Existing
rank-zero helpers, strict model loading, real tiny Safetensors shard tracing,
VLM/DiT values, legacy sensor formats, and mapped sensor/RNG parity are covered.

Measured fresh-process **32 MiB synthetic ZIP checkpoint**, warm page cache:

| Measurement | Eager | mmap |
| --- | ---: | ---: |
| Load elapsed | 0.02095 s | 0.00161 s |
| Load + touch all tensor pages | 0.02469 s | 0.00766 s |
| RSS immediately after load | 549,864 KiB | 517,032 KiB |
| Peak RSS after all pages touched | 551,528 KiB | 551,468 KiB |
| Load `rchar` delta | 33,577,769 B | 26,891 B |
| Physical `read_bytes` delta | 0 B | 0 B |

Checksums match. This demonstrates removal of eager read/copy traffic in that
small test. It does **not** demonstrate a material full-model peak-RAM decrease
or fewer physical disk reads; after touching all pages, peaks are almost equal.
It is not a cold PM893 SSD benchmark. No system cache flush was performed.

Four-GPU **loading-only** validation is provided separately, using the production
model factory without creating datasets, Trainer, an optimizer, or an engine:

```bash
GROOT_DEBUG_MODEL_LOAD_IO=1 GROOT_MODEL_LOAD_TRACE_DIR=/tmp/load-smoke-trace \
GROOT_SMOKE_VERIFY_DIT=1 GROOT_PATCH_MISTRAL=1 GROOT_SKIP_HF_MODEL_WEIGHTS=0 \
NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/smoke_distributed_model_loading.py \
  model=gr00t_sidecar hardware=quad_3090_zero3 \
  model.base_model_path=/path/to/GR00T-N1.7-3B \
  model.point_encoder_checkpoint_path=/path/to/concerto_small.pth \
  model.tactile_pretrained_model=/path/to/sparsh \
  output_dir=/tmp/fresh-loading-smoke-output
```

This deliberately stops **before** existing TrainingArguments/engine setup;
it validates loading for the four-GPU ZeRO-3 training configuration, not a
ZeRO-3 engine step. Actual engine validation must then use the unchanged Hydra
training entry point with the same dataset/hardware and `training.max_steps=2`.
No full-model rank-0 streaming or earlier ZeRO initialization is used in either.

**Executed on the host:** four-GPU NCCL process-group loading with Gloo control,
GR00T + Concerto Small + SPARSH, exact original DiT audit on every rank, and a
second four-GPU run isolating eager versus mapped sensor reads. The sandbox
hid GPUs, so host execution was required. The first failed offline attempt also
proved that a catchable loader error propagated to all ranks and terminated
without an NCCL watchdog hang. Existing `GROOT_PATCH_MISTRAL=1` was used equally
for the offline Qwen tokenizer in subsequent runs; no tokenizer code was changed.

**Not executed:** actual NCCL watchdog-timeout reproduction, a ZeRO-3 engine
initialization/step, full UniVTAC two-step forward/backward, and a controlled
cold-SSD benchmark. The supplied/default `data/univtac` dataset directory was
not present; no different dataset was substituted. The loading test stops before
TrainingArguments exactly as production currently loads its model. Full engine
and optimizer behavior therefore remain a required follow-up validation.

The user's hardware YAML was edited during the task and contained unsupported
`hardware.batch_size`/`hardware.gradient_accumulation_steps` aliases. Repository
YAML files were not modified. Host validation used a **temporary loading-only
snapshot** with all interpolations resolved first, retaining the resolved training
values (batch 32, accumulation 32) and omitting only the unsupported hardware
aliases. No training batches or optimizer steps ran. The production schema issue
is not silently fixed by this loading optimization.

### Actual four-GPU measurements

Full bounded records, filenames/sizes, per-shard events, and integrity results:
[`distributed_checkpoint_loading_measurements.json`](distributed_checkpoint_loading_measurements.json).
The earlier FP32-storage attempt failed; its numbers are incomplete records,
not a complete matched before/after experiment.

| Rank | Before: GR00T end RSS (KiB) | BF16 preserved: GR00T end RSS (KiB) | BF16 loading-complete peak (KiB) |
| --- | ---: | ---: | ---: |
| 0 | 9,076,364 | 1,458,556 | 2,493,964 |
| 1 | 8,927,224 | 1,560,936 | 2,454,176 |
| 2 | did not complete | 1,933,476 | 2,408,240 |
| 3 | did not start | 1,551,724 | 2,446,332 |

The candidate completed serialized loading on every rank in about **41.615 s**.
Those peaks precede the separate numerical audit, which touches mapped DiT pages
and raises RSS; mapped VLM pages can also fault later at engine/device transfer.
Do not read these numbers as final training RAM or as all weights already being
physically resident.

A safe second run kept BF16 preparation and Gloo unchanged and disabled only
sensor mmap. Both four-rank runs passed strict loading:

| Rank | Concerto eager `rchar` delta | Concerto mmap `rchar` delta | Eager read time | mmap read time |
| --- | ---: | ---: | ---: | ---: |
| 0 | 154,815,072 B | 631,544 B | 0.3132 s | 0.0382 s |
| 1 | 154,815,064 B | 631,515 B | 0.1084 s | 0.0209 s |
| 2 | 154,815,049 B | 631,532 B | 0.0984 s | 0.0207 s |
| 3 | 154,815,048 B | 631,520 B | 0.0961 s | 0.0211 s |

This removes about **616.7 MB** of eager userspace read/copy traffic over four
ranks in that sensor comparison. Whole loading was 42.860 s eager versus 41.615 s
mmap in these single sequential runs; no statistically established whole-run
speedup is claimed. Whole-process loading peaks stayed around 2.4–2.5 million
KiB, dominated by construction, so mmap alone did not establish a peak-RAM gain.
Page-cache histories differed: physical reads for whole loading were 153,387,008 B
on eager rank 0 versus 193,572,864 B on mmap rank 0 (other ranks mostly zero).
Therefore **no reduction in total physical SSD I/O is claimed**. Pages can simply
be read later, as demonstrated by the separate DiT audit.

No actual NCCL timeout occurred during these runs. The memory-constrained run
terminated with SIGKILL, and the corrected loading runs completed. The original
production watchdog failure is not claimed to be reproduced or definitively cured.


### Recorded CPU results

The final focused suite passed **48 tests**, skipped the opt-in real-checkpoint unit integration and
excluded one GPU-marked test. The existing distributed-helper suite plus the
new four-rank cases passed **12 tests**; the final four-rank rerun passed both
success/failure cases again. Ruff and `git diff --check` passed.

For the final four-CPU-rank successful smoke test (32 floating values, **not**
the production model), the synthetic 0.1 s delay was on rank 2:

| Rank | Process peak RSS (KiB) | Own loader body (s) | Status-reduction wait, total (s) | Physical read_bytes in body |
| --- | ---: | ---: | ---: | ---: |
| 0 | 710,936 | 0.005139 | 0.127033 | 0 |
| 1 | 712,116 | 0.005177 | 0.128388 | 0 |
| 2 | 712,428 | 0.107066 | 0.027765 | 0 |
| 3 | 710,832 | 0.004982 | 0.126505 | 0 |

Loader intervals did not overlap. Every rank received the exact stored tensor;
rank-2 failure was propagated to all ranks in the separate failure case, with
rank 3's loader never invoked. These numbers demonstrate control/wait tracing
and serialization, not production speedup, physical host-memory savings, or
an actual NCCL timeout reproduction.

### Files changed for this optimization

- `gr00t/utils/dist_utils.py`: CPU control group and collective tracing.
- `gr00t/utils/model_load_diagnostics.py`: per-rank RSS/I/O/time/file diagnostics.
- `gr00t/utils/checkpoint_io.py`: PyTorch ZIP mmap with legacy-format support.
- `gr00t/model/gr00t_n1d7/setup.py`: trace scopes, BF16 metadata selection, and serialization safety guard.
- `gr00t/model/gr00t_n1d7/pretrained_loading.py`: header dtype/shape validation and selective meta dtype preparation.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: apply verified meta dtype preparation and pass the existing optimization option to deferred sensors.
- `gr00t/model/extension/point_encoder.py`: mapped official checkpoint read, preserving construction/precision.
- `gr00t/model/extension/tactile_encoder.py`: mapped Torch checkpoint read; Safetensors unchanged.
- `gr00t/experiment/experiment.py`: one diagnostic marker before the existing TrainingArguments call.
- `scripts/benchmark_checkpoint_io.py`, `scripts/summarize_model_load.py`, `scripts/smoke_distributed_model_loading.py`.
- Focused tests: `test_checkpoint_io.py`, `test_serialized_loading.py`, and mapped/shard-trace regressions in the existing Concerto, SPARSH, and pretrained-loading tests.
- This report and its bounded measurement JSON. Earlier unrelated working-tree changes were preserved.
