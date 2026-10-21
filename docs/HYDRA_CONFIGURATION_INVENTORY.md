# Configuration inventory (before Hydra implementation)

Inspected the working tree, including existing uncommitted changes. The legacy path is
shell → Tyro FinetuneConfig → launch_finetune assignments → Config → experiment.run →
Gr00tN1d7Pipeline → AutoModel/AutoProcessor → TrainingArguments → Gr00tTrainer.
launch_train also accepts internal Config directly and reapplies Tyro after file loading.

## Complete fine-tuning CLI field inventory

```text
base_model_path
dataset_path
embodiment_tag
modality_config_path
seed
dit_type
use_point_conditioning
use_tactile_conditioning
point_input_dim
tactile_input_channels
tactile_encoder_cfg
tactile_pretrained_model
tactile_checkpoint_filename
tactile_background_path
point_encoder_cfg
point_encoder_checkpoint_path
point_encoder_repo_id
point_encoder_download_root
point_encoder_grid_size
point_encoder_enable_flash
tune_llm
tune_visual
tune_projector
tune_diffusion_model
use_lora
lora_r
lora_alpha
lora_dropout
lora_bias
tune_vlln
tune_point_encoder
tune_tactile_encoder
tune_multimodal_adapter
state_dropout_prob
random_rotation_angle
color_jitter_params
disable_color_jitter
use_percentiles
shortest_image_edge
crop_fraction
extra_augmentation_config
global_batch_size
dataloader_num_workers
learning_rate
vlm_learning_rate
action_head_learning_rate
point_encoder_learning_rate
tactile_encoder_learning_rate
action_head_dropout
vl_self_attention_dropout
gradient_accumulation_steps
output_dir
experiment_name
wandb_project
save_steps
save_total_limit
num_gpus
deepspeed_stage
zero3_offload_optimizer
zero3_offload_param
zero3_sub_group_size
gradient_checkpointing
use_wandb
telegram_on
telegram_chat_id
telegram_notify_start
telegram_notify_save
telegram_notify_finish
telegram_notify_error
discord_on
discord_notify_start
discord_notify_save
discord_notify_finish
discord_notify_error
max_steps
weight_decay
warmup_ratio
ds_weights_alpha
shard_size
episode_sampling_rate
allow_padding
num_shards_per_epoch
save_only_model
resume_from_checkpoint
skip_weight_loading
enable_training_dashboard
dashboard_refresh_seconds
enable_multimodal_diagnostics
multimodal_diagnostics_interval
multimodal_diagnostics_level
```

## Effective ownership and discrepancies

| Layer | Effective behavior |
|---|---|
| Shell full multimodal | 5000 steps, batch 2 × accumulation 16, two GPUs/ZeRO3, all four LRs 1e-4, episode rate 1; frozen VLM, train head/sensors/adapters; current working tree already passes point dimension **6** |
| Shell UniVTAC overfit | episodes 0..4 in independent symlink subset, recomputed stats, 5000 steps, batch 2 × 1, LR 3e-4, no weight decay/warmup/state/head/self-attention dropout/jitter, crop fraction 1, full min/max normalization; evaluates the same five episodes after training |
| Duplicate defaults | FinetuneConfig vs TrainingConfig: batch 64 vs 1024, steps 10000 vs 30000, sensor LR 1e-5 vs None; model state dropout .8 vs launcher .2, point dimension 3 vs CLI 6, pointnet2 vs concerto_small; processor crop defaults also differ |
| Launcher constants | fused AdamW, load_bf16=False, reproject_vision=False, Cosmos model name, FP32 trainable backbone, use_relative_action=True; fractional resize clears legacy crop/target sizes |
| AutoModel explicit overrides | tuning flags, state dropout, load dtype, DiT selector, conditioning/dimensions, encoder options, temporal offsets, optional dropout overrides; LoRA enabled after base loading or inherited if already present |
| Checkpoint architecture | hidden widths, layer counts, diffusion_model_cfg, max dimensions/horizon, pretrained embedded sensor architecture and weights; not replaced wholesale by internal dataclass defaults |
| Sensor loading | embedded sensors validated; newly enabled encoders deferred and bootstrapped after base load; mismatch checking rejects unexpected/missing pretrained weights |
| AutoProcessor overrides | modalities, percentile/relative normalization, all image transforms, model name/type, action maximum horizon from internal model config, state encoding; processor state dropout forced to zero (action head applies it once) |
| Not exposed by fine-tune CLI | optimizer selection, scheduler, clipping, warmup_steps, evaluation split/strategy, shuffle/subsample/multiprocessing, precision flags, local-only checkpoint loading, transformer cache, most architecture fields; internal Config still exposes these |
| Batch semantics | global_batch_size is before accumulation; effective optimizer batch = global batch × accumulation; per_gpu_batch_size is deprecated and overrides global batch |
| Optimizer | identity-based VLM/head groups plus optional sensor backbone groups, decay/no_decay split; sensor projections/adapters remain in head groups; sensor LR fallback is head LR only when unset |
| Resume | shell checks trainer/optimizer/scheduler/RNG/model state (including DeepSpeed partitions); Trainer checks state again; action semantics verified before rewriting run config; restores counters, optimizer, scheduler and RNG, so requested LR is not necessarily restored LR |
| Logging | model dataclass dump is requested config, not checkpoint-effective architecture; final_model_config.json is authoritative after loading; final_processor_config.json records processor overrides; optimizer logs already report actual tensor ownership |
| UniVTAC contract | processor requires pointcloud.scene XYZRGB [N,6], tactile.rgb RGB; Sparsh requires [previous,current] offsets [negative,0]; current shared Python modality builder uses dataset horizon 16 (model maximum 40) |
| SO100 | examples/SO100/so100_config.py uses front/wrist, single_arm/gripper, horizon16, relative arm/absolute gripper; demo_data/cube_to_bowl_5 is existing script dataset |

Additional inspected paths: gr00t/data/types.py, model/extension/{point_encoder,tactile_encoder}.py,
processing_gr00t_n1d7.py, torchrun_supervisor.py, SO100 script/config, existing resume tests,
open_loop_eval.py invocation in overfit workflow. Existing notifications use environment-only secrets.

Hydra will preserve setup.py loading and optimizer mathematics. It will reject unsupported
architecture changes rather than imply they are applied. A 40-step multimodal YAML recipe
is an intentional difference from the current 16-step shared Python modality default;
users can override it to 16. RGB/overfit retain 16.

Further factory audit during implementation: `DatasetFactory` asserts eval_strategy=no and
mode=single_turn. The legacy eval split, upload/RL/open-loop toggles, eval_bf16,
DataConfig shuffle/subsample/random_chop/mock/download/image size fields are not read by
this factory/experiment. Hydra rejects these ineffective controls rather than printing
that they are applied. Separate open-loop evaluation remains supported. Trainer forces
ignore_data_skip=True; resume restores state, but does not promise exact data-stream replay.
