# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    Gr00tN1d7Processor,
)
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


def log_parameter_summary(model: Gr00tN1d7) -> None:
    """Log and validate the trainable split used by backbone-LoRA fine-tuning."""
    named_parameters = list(model.named_parameters())
    total_params = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_params = sum(
        parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
    )
    lora_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("backbone.") and "lora_" in name and parameter.requires_grad
    ]
    action_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("action_head.") and parameter.requires_grad
    ]
    frozen_backbone_base = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("backbone.") and "lora_" not in name and not parameter.requires_grad
    ]

    logging.info("Total parameters: %s", f"{total_params:,}")
    logging.info(
        "Trainable parameters: %s (%.2f%%)",
        f"{trainable_params:,}",
        100 * trainable_params / total_params,
    )
    logging.info("LoRA trainable parameters: %s", f"{sum(p.numel() for _, p in lora_parameters):,}")
    logging.info(
        "Action-head trainable parameters: %s",
        f"{sum(p.numel() for _, p in action_parameters):,}",
    )
    logging.info(
        "Frozen backbone base parameters: %s",
        f"{sum(p.numel() for _, p in frozen_backbone_base):,}",
    )

    if model.config.use_lora:
        if not lora_parameters:
            raise RuntimeError("LoRA is enabled but zero LoRA parameters are trainable")
        if not action_parameters:
            raise RuntimeError(
                "LoRA is enabled but the GR00T action head has no trainable parameters"
            )

        trainable_backbone_base = [
            name
            for name, parameter in named_parameters
            if name.startswith("backbone.") and "lora_" not in name and parameter.requires_grad
        ]
        if trainable_backbone_base:
            raise RuntimeError(
                "LoRA mode requires every backbone base parameter to be frozen; found: "
                f"{trainable_backbone_base[:5]}"
            )

        required_action_modules = [
            "action_head.state_encoder.",
            "action_head.action_encoder.",
            "action_head.action_decoder.",
            "action_head.model.",
        ]
        if model.config.add_pos_embed:
            required_action_modules.append("action_head.position_embedding.")
        if model.config.use_vlln:
            required_action_modules.append("action_head.vlln.")
        missing_action_modules = [
            prefix
            for prefix in required_action_modules
            if not any(name.startswith(prefix) for name, _ in action_parameters)
        ]
        if missing_action_modules:
            raise RuntimeError(
                "LoRA mode requires the complete action head to remain trainable; "
                f"no trainable parameters found under {missing_action_modules}"
            )

        logging.info("Representative trainable LoRA parameter: %s", lora_parameters[0][0])
        if frozen_backbone_base:
            logging.info(
                "Representative frozen backbone base parameter: %s", frozen_backbone_base[0][0]
            )
        for prefix in required_action_modules:
            representative = next(name for name, _ in action_parameters if name.startswith(prefix))
            logging.info(
                "Representative trainable %s parameter: %s",
                prefix.rstrip("."),
                representative,
            )
        logging.info(
            "NEW_EMBODIMENT uses existing projector/category ID %d",
            EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"],
        )


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            other_missing = [k for k in missing_keys if "mask_token" not in k]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

            if self.config.model.use_lora and not model.config.use_lora:
                model.enable_lora(
                    r=self.config.model.lora_r,
                    alpha=self.config.model.lora_alpha,
                    dropout=self.config.model.lora_dropout,
                    bias=self.config.model.lora_bias,
                )
            elif model.config.use_lora:
                logging.info("Loaded LoRA structure and weights directly from checkpoint metadata")

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(f"Model Config: {model.config}")
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                    f.write(model.config.to_filtered_json())
        log_parameter_summary(model)
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        letter_box_transform = self.model_config.letter_box_transform
        logging.info("N1.7 letter_box_transform=%s", letter_box_transform)
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        with run_or_wait_on_rank0(label="final_processor_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                    json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                stats_dict = convert_tensors_to_lists(stats)
                with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                    json.dump(stats_dict, f, indent=2)
                logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
