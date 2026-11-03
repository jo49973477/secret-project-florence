# Historical training launchers

These scripts are retained for reproducibility. They are not the preferred
entrypoints for new Florence experiments. Invoke them from the repository root
as `bash scripts/legacy/<name>.sh`; their repository-root and shared-library
paths have been adjusted for this directory.

`launch_univtac_controlled_experiment.sh` uses an older **A/B/C** comparison
(RGB baseline and sensor variants). The current Florence **A/B/C/D** modes in
`scripts/train_florence_ablations.sh` compare Temporal or MoE conditioning with
DiT LoRA or full DiT tuning on the full UniVTAC dataset. The letters are not
interchangeable between the two launchers.

`finetune_univtac_lora.sh` is a historical **VLM LoRA** launcher. Florence's
`A` and `B` modes use **DiT LoRA**. The remaining scripts here are early
UniVTAC training and evaluation experiments.
