# GR00T N1.7 Multimodal Extension Review Report

## 1. Existing Data Pipeline

Before these changes, the dataset half of the pipeline already worked:

- UniVTAC tactile originates from the selected HDF5 tactile RGB stream and is decoded as `tactile.rgb`.
- Point clouds originate from depth plus camera intrinsics and are stored as fixed-size `pointcloud.xyz`.
- `LeRobotEpisodeLoader` places both in episode DataFrame columns.
- `extract_step_data()` preserves them in `VLAStepData.tactile` and `VLAStepData.pointclouds`.
- `DatasetFactory` and `ShardedSingleStepDataset` required no architectural changes.

The break was later in the pipeline:

- `Gr00tN1d7Processor` discarded tactile and point-cloud values.
- The collator would have retained arbitrary numeric fields, but the processor never emitted them.
- `Gr00tN1d7.prepare_input()` already moved arbitrary floating tensors to the model device and dtype.
- `Gr00tN1d7ActionHead.prepare_input()` already preserved arbitrary batch keys.
- Therefore, before this fix, `action_input.points` and `action_input.tactile` did not exist.
- Direct policy inference also discarded both while unbatching observations.

After the fix, the model-facing shapes are:

```text
UniVTAC pointcloud.xyz: [1, 1024, 3] float32
processor output:       [1024, 3]
collated/action_input:  [B, 1024, 3]

UniVTAC tactile.rgb:    [1, H, W, 3] uint8
processor output:       [3, H, W] float32 in [0, 1]
collated/action_input:  [B, 3, H, W]
```

The current UniVTAC data contains XYZ only. XYZRGB is supported by configuring multiple aligned point fields and setting `point_input_dim=6`; no RGB values are fabricated.

## 2. Data-Pipeline Fixes

The smallest missing paths were added in:

- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`: converts point fields to model-ready `[N, D]` tensors and tactile HWC `uint8` to CHW `float32`.
- `gr00t/policy/gr00t_policy.py`: preserves optional sensors during direct-policy unbatching and places them in `VLAStepData`.
- `gr00t/policy/gr00t_policy.py`: validates optional point-cloud/tactile shape, dtype, horizon, and batch size.

No collator rewrite was necessary. Its existing generic stacking path produces `action_input.points` and `action_input.tactile`.

## 3. Inference Caching

`Gr00tN1d7ActionHead._encode_multimodal_inputs()` now owns sensor encoding.

`get_action_with_features()` invokes it once before the denoising loop and reuses:

```python
point_tokens
point_attention_mask
tactile_tokens
```

for every flow-matching step. A call-count regression test confirms each encoder runs exactly once with three denoising iterations.

## 4. Train/Freeze Behavior

The following explicit flags were added:

```python
tune_point_encoder
tune_tactile_encoder
tune_multimodal_adapter
```

The requested configuration now produces:

```text
base DiT                         frozen
point encoder                    trainable
tactile encoder                  trainable
point/tactile cross-attention    trainable
point/tactile gates              trainable
```

`MultiModalConditionedDiT.set_multimodal_adapter_trainable()` controls the adapter modules directly without parameter-name matching.

Frozen point/tactile encoders are placed in evaluation mode, including their BatchNorm and Dropout layers. Adapter Dropout mode is also controlled independently when the base DiT is frozen.

The standard fine-tuning configuration now exposes the DiT selector, sensor switches, encoder choice/input dimensions, and all freeze flags through `gr00t/configs/finetune_config.py` and `gr00t/experiment/launch_finetune.py`.

## 5. Conditional Module Instantiation

`Gr00tN1d7ActionHead` now initializes both sensor encoders to `None`.

They are instantiated only for:

```python
dit_type == "multimodal_conditioned_dit"
```

and only when their corresponding conditioning flag is enabled.

The following configurations were validated:

- Original `AlternateVLDiT`: no sensor encoders.
- Original `DiT`: no sensor encoders.
- Point only.
- Tactile only.
- Point and tactile together.

The legacy `use_alternate_vl_dit` selector remains compatible with old checkpoint configurations.

## 6. AlternateVLDiT Behavior

`MultiModalConditionedDiT` now extends `AlternateVLDiT`.

Its original scheduling is preserved:

- Odd blocks perform state/action self-attention.
- Cross-attention blocks alternate between non-image/text tokens and image tokens.
- `image_mask` and `backbone_attention_mask` are required and passed during training and inference.
- Point/tactile cross-attention is applied afterward, only to the action-token suffix.
- Original timestep conditioning, output normalization, and projection remain unchanged.

With zero-initialized gates, its output matches an equivalent `AlternateVLDiT` exactly in the sanity test using `rtol=0` and `atol=0`.

## 7. Validation Performed

Results:

```text
Full tests/gr00t/model:
97 passed, 7 skipped

Dataset factory + sharded dataset + UniVTAC multimodal:
29 passed

Direct policy tests:
10 passed

Focused multimodal/model/policy suite:
61 passed
```

Observed processor/collator shapes:

```text
points:
  sample    (32, 3)
  batch     (2, 32, 3)

tactile:
  sample    (3, 24, 32)
  batch     (2, 3, 24, 32)
```

Additional checks covered:

- Point-only, tactile-only, and combined training forward passes.
- Independent base-DiT/adapter freezing.
- Frozen encoder evaluation mode.
- Device and dtype movement in `Gr00tN1d7.prepare_input()`.
- Single encoder execution during inference.
- Exact zero-gate equivalence.
- Legacy DiT selector compatibility.
- Ruff lint/format checks and `git diff --check`.
- Fine-tuning CLI option generation.

The related UniVTAC README was updated with the experimental multimodal fine-tuning command and current limitations.

## 8. Remaining Integration Risks

- No full 3B checkpoint GPU fine-tuning or open-loop evaluation was run; validation used lightweight model configurations.
- UniVTAC currently supplies XYZ, so its actual action input is `[B, 1024, 3]`, not XYZRGB `[B, 1024, 6]`.
- The model processor currently supports one tactile/point-cloud timestep. The dataset loader supports arbitrary `delta_indices`, but multi-frame sensor aggregation needs an explicit model-side design.
- Point coordinates remain in the selected depth-camera frame and are not normalized through dataset statistics.
- Tactile MP4 storage is lossy; processor normalization only scales decoded pixels to `[0, 1]`.
- With gates initialized exactly to zero, the first optimization step primarily trains the gates; encoder/cross-attention gradients become effective once gates move away from zero.
- `Gr00tSimPolicyWrapper` still represents its legacy flat RGB/state/language interface. Direct `Gr00tPolicy` supports the new nested modalities.
