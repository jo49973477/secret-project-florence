# Concerto sparse-convolution precision boundary

The installed spconv 2.3.8 tensorview mapping supports FP16/FP32 but not BF16.
DeepSpeed BF16 training stores Concerto's sparse weights **and bias** in BF16.
Nested FP16 autocast does not change parameter storage. Frozen sensor backbones
are put in eval mode by the existing action-head logic. In spconv's
`SparseConvolution._conv_forward`, that eval branch calls `ops.implicit_gemm`
directly, bypassing the `custom_fwd(cast_inputs=float16)` function used by the
training branch. Its fused bias reaches `torch_tensor_to_tv` as BF16 and raises
`KeyError: torch.bfloat16`. Weights and output-dtype selection are also outside
that training-only cast. This is why a train-mode-only autocast test missed the
frozen-backbone failure.

`ConcertoPointEncoder` now adapts each of its sparse convolution instances at
`_conv_forward(training, input, weight, bias, ...)`. This is a local operation
boundary; no spconv source files, global functions, or dtype mappings are changed
by the training implementation. It preserves the original forward/module hooks,
state-dict names, training/eval flags, parameter identity, and parameter values.
The adapter checks the expected spconv method signature and fails clearly if a
future version changes this private API.

## Exact policy

| Tensor/operation | Precision |
| --- | --- |
| Geometry preprocessing / coordinates / initial feature assembly | FP32, outside autocast |
| Grid coordinates | INT32 |
| Offsets / batch indices | INT64 |
| Dense Concerto CUDA operations | Existing local FP16 autocast |
| Registered sparse weight / bias | Existing storage dtype; BF16 under BF16 ZeRO-3 |
| SparseConvTensor features at the convolution boundary | Explicit FP16 on CUDA |
| Weight / bias arguments supplied to sparse convolution | Differentiable FP16 casts on CUDA |
| Optional sparse residual | Explicit FP16 on CUDA |
| CUDA sparse convolution output | FP16 |
| Dense projection input | Cast to registered projection-weight dtype, BF16 in the reported setup |
| Main GR00T / Sidecar BF16 training | Unchanged |

CPU operation-boundary tests use FP32 arguments with an emulated sparse kernel;
they do not claim CPU support for spconv's fused inference-bias kernel. No GPU
storage or global precision setting is changed. Each sparse call disables
ambient autocast locally after explicitly converting its arguments, so both
training and inference paths see supported, matching types. There is no
parameter replacement, `.data` mutation, detachment, cast cache, or modification
of checkpoint loading. Autograd propagates through `.to(dtype=...)` back into
the original registered parameters. Pretrained stored values remain unchanged
by forward/backward; normal optimizer updates are still permitted.

The existing ZeRO-3 `PointSequential` CPE leaf registration and synchronization
of `Point.feat` with `SparseConvTensor.features` are retained. ZeRO gathers the
original parameters through the existing module hooks before the operation
casts them. The differentiable feature cast retains the leaf's backward-hook
path. No DeepSpeed initialization timing or production optimizer is changed.

## Trace and validation

Enable `GROOT_DEBUG_CONCERTO_PRECISION=1` or `GR00T_DTYPE_DEBUG=1`. Logs include
initial features, geometry/index types, and each sparse convolution's source
and compute feature/weight/bias types, its training flag, and its output type.
Projection logs record the transition back to the projection's dtype. These
logs are emitted on the first successful call per sparse module.

Focused tests:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m pytest \
  tests/gr00t/model/test_concerto_precision.py \
  tests/gr00t/model/test_concerto_checkpoint_loading.py \
  tests/gr00t/experiment/test_training_notification_callback.py -q
```

The CPU tests reproduce the unsupported BF16 bias without the boundary, then
exercise two optimizer steps with trainable and frozen backbones, biased and
bias-free convolutions, residual casts, finite gradients, unchanged state-dict
keys/storage/parameter identities, idempotent setup, and retained module hooks.
The CUDA tests use real `spconv.SubMConv3d` with BF16 parameter storage and check
kernel feature/weight/fused-bias/output types in both trainable and frozen modes.
Scoped test/diagnostic spies observe kernel calls; they never alter dtype maps.

`scripts/diagnose_concerto_precision.py` prints installed mappings without
loading a model when called without point data. With a local official Concerto
Small checkpoint and a real `[N, 6]` UniVTAC XYZRGB `.npy` scene, it can run two
steps with BF16 storage and finite output/gradient checks:

```bash
GROOT_DEBUG_CONCERTO_PRECISION=1 .venv/bin/python scripts/diagnose_concerto_precision.py \
  --points-npy /path/to/scene.npy --checkpoint-path /path/to/concerto_small.pth --steps 2

GROOT_DEBUG_CONCERTO_PRECISION=1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/diagnose_concerto_precision.py \
  --points-npy /path/to/scene.npy --checkpoint-path /path/to/concerto_small.pth \
  --steps 2 --zero3
```

Repeat with `--frozen-backbone` to exercise eval-mode fused bias. The standalone
harness uses a diagnostic SGD optimizer and BF16 ZeRO-3, without optimizer or
parameter offload; it does not change the production training optimizer. Its
ZeRO-3 path checks full gradients collectively before stepping and requires a
local checkpoint to avoid concurrent downloads. It does not load the 3B GR00T
model or substitute for a full UniVTAC two-step training validation.

Local validation: 42 tests passed and two CUDA cases were skipped. GPU execution
was unavailable (`torch.cuda.is_available() == False`, zero CUDA devices, no
`/dev/nvidia*`, and `nvidia-smi` could not communicate with the driver). Therefore
real Concerto CUDA forward/backward, two-rank ZeRO-3 execution, and a successful
full UniVTAC two-step training smoke test remain **unverified**. CPU emulation is
not evidence of those GPU results. FP16 dynamic-range behavior with real inputs
also requires the target-host finite-gradient checks.

The failure-notification callback also now imports `os` for its atomic delivery
marker. Tests use a mock sender to verify successful marker creation, duplicate
suppression, and handling of marker-write errors without masking training's
original failure. No real notifications are sent by the tests.
