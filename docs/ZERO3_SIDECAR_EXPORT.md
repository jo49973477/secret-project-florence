# SPL36 WAH MoE Sidecar + DiT LoRA export

The original exporter required a Hugging Face `config.json` in a native DeepSpeed
checkpoint. Native ZeRO-3 saves intentionally skip HF consolidation. They instead
contain rank shards and training provenance. It also passed a filename to the
DeepSpeed 0.17.6 converter's **directory** argument, producing a nested directory,
and failed to copy the root processor configuration, normalization statistics,
and embodiment mapping.

Model construction lives in `gr00t/model/gr00t_n1d7/setup.py`: the original
pretrained GR00T architecture is loaded, Sidecar modules are added, and
`SidecarConditionedDiT` inserts DiT LoRA and the contact input MoE. Backbone LoRA
(`use_lora`) and DiT LoRA (`dit_training_mode=lora`) are separate settings. The
resolved `experiment_cfg/final_model_config.json` is written after construction;
it captures inherited DiT dimensions, layer selection, embedded sensor architecture,
MoE routing, and LoRA settings. The exporter promotes this file to `config.json`.
An optional `--model-config-dir ORIGINAL_PRETRAINED_DIRECTORY` merges the original
config beneath these authoritative resolved settings. No original pretrained
weights should be overlaid on restored training weights.

`Trainer._save_checkpoint` saves native DeepSpeed partitions; the experiment save
callback copies experiment configuration and processor files into each checkpoint.
Frozen model weights live in `frozen_param_fragments` in the large model shards;
small optimizer shards contain the trainable FP32 partitions. Both are required.
The installed DeepSpeed converter supports this layout and defaults to including
frozen parameters; the exporter makes that choice explicit.

Inspection of this checkpoint on 2026-11-06 passed: two ZeRO-3 partitions, 1,907
parameter entries, 87 Sidecar, 256 DiT LoRA, and 12 contact input MoE entries.
Rank layouts, trainable partition sizes, frozen fragment sizes, and persistent
buffer presence were checked. This establishes structural completeness; full
conversion and numerical restoration remain separate checks.

## SPL36 commands

Run from the repository checkout. Inspection does not consolidate weights:

```bash
cd /home/spl_yeongyoo/secret-project-wah
source .venv/bin/activate
export CKPT=outputs/moe_sidecar_ditlora-full/20261103_224803/checkpoint-10000
export EXPORT=exports/moe_sidecar_ditlora-full-step10000
python scripts/export_zero3_checkpoint.py "$CKPT" --inspect
```

The following command is the expensive export; run only when explicitly authorized.
Budget host RAM for full FP32 weights, a loaded inference model, and verification.
The destination must not exist. Nothing writes into the native checkpoint.

```bash
PYTHONPATH=. python scripts/export_zero3_checkpoint.py "$CKPT" "$EXPORT"
```

Export writes sharded safetensors, resolved config, processor configuration,
statistics, embodiment mapping, processor directory when present, and provenance.
It then loads `Gr00tN1d7.from_pretrained`, rejects missing, unexpected, mismatched,
and loading-error entries, and compares **every Sidecar, MoE, and LoRA tensor**
with native reconstruction after conversion to the loaded tensor's dtype.
Only successful verification writes `export_verification.json`. A failed export
may leave a partial destination; retain it for diagnosis and choose a new output
path for retrying. The command never overwrites an existing destination.

Repeat the full native comparison independently:

```bash
PYTHONPATH=. python scripts/export_zero3_checkpoint.py "$CKPT" "$EXPORT" --verify-only
TEST_CACHE_PATH=/tmp/wah-test-cache python -m pytest tests/scripts/test_export_zero3_checkpoint.py -q
```

Original Cosmos model/tokenizer assets must remain available in the HF cache or
via authenticated Hub access: the current inference constructor/processor can
still consult `nvidia/Cosmos-Reason2-2B`. Export embeds trained sensor weights and
their architecture; it does not vendor that external tokenizer/model cache.
If resolved `final_model_config.json` is missing, export fails instead of guessing
from Hydra defaults. Recover that file from the run before exporting.

## Transfer to RedForce

These commands assume `RedForce` is your SSH alias and its checkout is
`~/secret-project-wah`; substitute the actual alias/path if different. Transfer
only after `export_verification.json` reports PASS. Run on SPL36:

```bash
test -f "$EXPORT/export_verification.json"
ssh RedForce 'mkdir -p ~/secret-project-wah/exports/moe_sidecar_ditlora-full-step10000'
rsync -av --checksum "$EXPORT/" RedForce:secret-project-wah/exports/moe_sidecar_ditlora-full-step10000/
```

Use matching repository code and dependencies on RedForce. Copy/cache the original
Cosmos assets separately if RedForce has no Hub access. Do not transfer native
optimizer shards for inference.

## Local smoke rollout on RedForce

In terminal 1, from the RedForce checkout with its GR00T environment activated:

```bash
cd ~/secret-project-wah
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_univtac_server.sh \
  --checkpoint "$PWD/exports/moe_sidecar_ditlora-full-step10000" \
  --embodiment-tag NEW_EMBODIMENT --device cuda:0 --host 127.0.0.1 --port 5555
```

In terminal 2, set the three machine-specific Isaac paths (they are not recorded
in this checkpoint), then run exactly this one-episode smoke:

```bash
cd ~/secret-project-wah
export UNIVTAC_ROOT=/absolute/path/to/UniVTAC
export UNIVTAC_PYTHON_COMMAND='/absolute/path/to/IsaacLab/isaaclab.sh -p'
export UNIVTAC_EXPECTED_PYTHON=/absolute/path/to/isaac-sim/python/bin/python3
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_univtac_rollout.sh \
  --checkpoint "$PWD/exports/moe_sidecar_ditlora-full-step10000" \
  --server-host 127.0.0.1 --server-port 5555 --mode smoke \
  --episodes-per-task 1 --tasks lift_can --execution-horizon 1 --device cuda:0 \
  --output-dir outputs/moe_sidecar_ditlora-step10000-smoke
```

Use a new result directory for subsequent runs. The expected Python path must be
the actual interpreter launched by IsaacLab. GPU inference, complete numerical
verification, simulator rollout, and the RedForce alias/Isaac paths remain to be
validated on their target hosts; no full export or rollout was run during repair.
