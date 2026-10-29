#!/usr/bin/env python
"""Four-GPU loading-only smoke test using the production pipeline and initialization order.

Launch explicitly with torchrun and existing Hydra model/hardware overrides.
Does not construct datasets, Trainer, TrainingArguments, optimizers, or an engine.
"""

import json
import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/univtac_full")
def main(cfg: DictConfig):
    from gr00t.configs.hydra_config import build_gr00t_config, inherit_checkpoint_architecture
    from gr00t.experiment.experiment import _init_distributed_process_group
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from gr00t.utils.dist_utils import run_or_wait_on_rank0, run_serialized_across_ranks
    from gr00t.utils.model_load_diagnostics import model_load_phase
    import torch
    from transformers import AutoConfig, set_seed

    if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS", "").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise SystemExit("Loading smoke must not use the test-only HF weight-skipping shortcut")
    config = build_gr00t_config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or config.training.num_gpus != 4:
        raise SystemExit("Use torchrun --nproc-per-node=4 and hardware=quad_3090_zero3")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise SystemExit("Four CUDA GPUs are required; this is not a CPU/Gloo substitute")
    if config.training.deepspeed_stage != 3 or not config.training.bf16 or config.training.fp16:
        raise SystemExit("This smoke test requires the existing BF16 ZeRO-3 configuration")
    if config.training.zero3_offload_optimizer or config.training.zero3_offload_param:
        raise SystemExit("CPU offload must remain disabled")
    # Same metadata inheritance as train_hydra_worker, before experiment.run's
    # existing process-group initialization. No ZeRO-3 setup is moved earlier.
    checkpoint = AutoConfig.from_pretrained(
        config.training.start_from_checkpoint,
        trust_remote_code=config.training.transformers_trust_remote_code,
        local_files_only=config.training.transformers_local_files_only,
        cache_dir=config.training.transformers_cache_dir,
        **({"revision": config.model.model_revision} if config.model.model_revision else {}),
    )
    inherit_checkpoint_architecture(cfg, config, checkpoint)
    rank = _init_distributed_process_group()
    set_seed(config.data.seed)
    output = Path(cfg.output_dir).resolve()
    with run_or_wait_on_rank0(label="loading smoke output") as is_rank0:
        if is_rank0:
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"Loading smoke requires a fresh output_dir: {output}")
            output.mkdir(parents=True, exist_ok=True)
    try:
        with model_load_phase("smoke.complete_load"):
            model = Gr00tN1d7Pipeline(config, output)._create_model()
        # No trainer/engine initialization: explicitly stop where production
        # finishes model loading, before its TrainingArguments/ZeRO-3 setup.
        report = {
            "rank": rank,
            "loading": model.pretrained_loading_report,
            "meta_parameters": [name for name, p in model.named_parameters() if p.is_meta],
        }
        if os.environ.get("GROOT_SMOKE_VERIFY_DIT", "0") == "1":
            from gr00t.model.gr00t_n1d7.dit_integrity import compare_shared_tensors
            from scripts.verify_pretrained_dit import CheckpointTensors

            def verify_dit():
                with model_load_phase("smoke.dit_integrity"):
                    return compare_shared_tensors(
                        CheckpointTensors(config.training.start_from_checkpoint), model.state_dict()
                    )

            report["dit_integrity"] = run_serialized_across_ranks(
                verify_dit, label="pretrained DiT integrity audit"
            )
            if report["dit_integrity"]["status"] != "PASS":
                raise RuntimeError(f"Rank {rank}: pretrained DiT integrity failed")
        (output / f"loading-smoke-rank-{rank}.json").write_text(json.dumps(report, indent=2))
        if report["meta_parameters"] or report["loading"]["status"] != "PASS":
            raise RuntimeError(f"Rank {rank}: checkpoint loading validation failed")
        print(f"rank={rank} loading-only smoke: PASS (ZeRO-3 engine/steps not run)", flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
