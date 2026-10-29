# GR00T N1.7 pretrained loading

For the subsequent four-rank I/O work, verified source-BF16 meta preparation,
actual host GPU measurements, and cgroup limit findings, see
[distributed checkpoint loading](distributed_checkpoint_loading.md). The small
CPU measurements in this document describe the earlier VLM-only experiment.

The training pipeline defaults to `model.optimize_pretrained_loading=true`.
Use `model.optimize_pretrained_loading=false` to retain the nested external
Cosmos pretrained weight load for comparison. Both paths retain strict
checkpoint compatibility checks and restore runtime VLM state after loading.
For example, append either override to an existing Hydra launch:

```bash
GROOT_DEBUG_MODEL_LOAD_MEMORY=1 python scripts/train_hydra.py \
  model=gr00t_sidecar model.optimize_pretrained_loading=true dry_run=true
```

`dry_run=true` validates configuration without constructing the model or running
training. Remove it only for an authorized training run with the appropriate
existing data and hardware overrides. Resume provenance checks remain strict;
changing this option does not bypass existing resume compatibility checks.

## Loading behavior and tradeoffs

Before model construction, the pipeline reads the GR00T Safetensors index and
actual shard headers using `safe_open` / `get_slice().get_shape()`. It does not
materialize checkpoint tensors for this check. The VLM architecture is built
from its external configuration, with the existing layer selection and dtype
settings. Every persistent VLM tensor must have a matching name and shape in
the GR00T checkpoint before the external VLM pretrained load can be skipped.
Tied aliases are checked conservatively; an omitted alias can cause a safe
fallback rather than assuming it can be recovered.

The structure inherits Transformers 4.57.3's enclosing `from_pretrained`
construction context: `init_empty_weights()` for ordinary loading, or the
already active DeepSpeed ZeRO-3 context. This avoids a separate full randomly
initialized CPU FP32 VLM. No extra meta context wraps the action head. The
complete GR00T checkpoint is still loaded by the existing Transformers loader,
with `low_cpu_mem_usage=True`, existing rank synchronization, and existing
missing/unexpected/mismatched-key validation.

Incomplete VLM coverage, absent Safetensors, and unsupported checkpoint formats
use the external pretrained VLM loader. Fallback does **not** waive missing-key
validation: a checkpoint missing required GR00T tensors can still fail the
existing compatibility check. Corrupt or inaccessible shard files raise errors.
Standalone `Qwen3Backbone` construction and direct GR00T construction keep their
external pretrained behavior. The optimization is selected by the training
pipeline, rather than trusting a saved configuration flag as proof of coverage.
Access to the external VLM **configuration** is still required.

After strict validation, missing tensors explicitly belonging to newly enabled
Sidecar adapters are initialized with the original small adapter constructors.
This is necessary because the generic HF missing-key initializer does not know
about standalone query parameters, point gates initialized to one, or action
and contact gates initialized to zero. Only missing adapter tensors are copied;
pretrained DiT, loaded Sidecar tensors, and sensor encoders are untouched.
Copies preserve target `requires_grad` and dtype. Under ZeRO-3, existing
partitioned parameters are gathered individually and broadcast from rank zero.
No optimizer or architecture is replaced.

RoPE buffers are recomputed after checkpoint loading, vision Conv3d memory
formats/hooks are restored idempotently, and remaining meta tensors cause a
hard error. Sensor bootstrap and validation retain their existing lifecycle.

DeepSpeed initialization timing, TrainingArguments, distributed initialization,
optimizer setup, and model parallelism are unchanged. There is no `device_map`.
In particular, ZeRO-3 may still materialize a full shard in Transformers'
loader; avoiding the redundant VLM load does not guarantee eliminating a RAM
peak dominated by GR00T shards. No checkpoint resharding or cache modification
is performed.

## Validation and memory measurement

Run the focused CPU tests with real tiny local Qwen3-VL checkpoints:

```bash
NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH=. .venv/bin/python -m pytest \
  tests/gr00t/model/test_pretrained_loading.py \
  tests/gr00t/model/test_sidecar_dit.py \
  tests/gr00t/model/test_dit_integrity.py \
  tests/gr00t/model/test_qwen3_rope.py -q -s
```

The new loading tests explicitly disable the test harness's default
`GROOT_SKIP_HF_MODEL_WEIGHTS` shortcut. They compare real checkpoint values,
legacy/optimized dtypes and freezing, forward/backward behavior, missing Sidecar
initialization, DiT integrity, sharded checkpoint loading, runtime buffers, and
absence of meta tensors. The memory comparison starts separate processes and
reports Linux `/proc/self/status` VmRSS and VmHWM in KiB. Tiny-model measurements
are dominated by Python/library imports and do not establish full-model RAM
savings.

`GROOT_DEBUG_MODEL_LOAD_MEMORY=1` logs RSS, high-water RSS, rank, and PID before
model construction, after VLM construction, before and after the enclosing
GR00T load, and at pipeline completion. The enclosing load's "before" log
precedes construction because HF combines construction and shard loading in one
call. VmHWM is a process-lifetime peak, not a sum across distributed ranks.

Full-model validation still requires an authorized pair of fresh runs with the
same checkpoint, hardware, data configuration, dtype, and seed, changing only
`model.optimize_pretrained_loading`. Retain diagnostics for every rank. Check
strict load validation and the existing Pretrained DiT Integrity Check, compare
VLM tensors against the source checkpoint after documented dtype conversion,
and run a forward/backward step on the existing two-GPU ZeRO-3 setup. Compare
per-rank peaks and simultaneous aggregate host RSS before claiming savings.
These GPU/full-model experiments are intentionally not run by the CPU tests.

## Local validation results

Validation used the installed Transformers 4.57.3, Accelerate 1.13.0, and
PyTorch 2.9.0+cu128; execution was CPU-only. The combined loading, Sidecar, DiT,
RoPE, and Hydra run passed 96 tests, skipped one, and excluded two unrelated
Hydra tests after an initial run exposed their incompatibility with existing
working-tree training settings (batch-size expectation and resume max_steps).
The final focused loader/Hydra suite passed 14 tests, including sharded loading,
runtime memory-format/RoPE checks, Sidecar/multimodal overrides, and checkpoints
whose VLM weights deliberately differ from the external VLM source.

A fresh-process tiny-fixture comparison measured:

| Path | Peak RSS (KiB) |
| --- | ---: |
| Legacy | 835,212 |
| Optimized | 834,928 |

This small difference is within the import-dominated fixture's run-to-run
variation and establishes **no meaningful memory improvement**. An earlier run
measured 834,692 versus 834,964 KiB, with the difference reversed. No full-size GR00T RAM
reduction, production checkpoint DiT audit, or two-GPU ZeRO-3 success is claimed.
Those remain required deployment validations. Sharding was tested for loading
correctness only; no production checkpoint sharding optimization was added.
