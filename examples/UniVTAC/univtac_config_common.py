# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared UniVTAC modality configuration builder."""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def build_univtac_config(*, multimodal: bool) -> dict[str, ModalityConfig]:
    """Build the stable RGB baseline, optionally adding sensor modalities."""
    config = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
        "action": ModalityConfig(
            # UniVTAC's ACT baseline uses 50, but N1.7 accepts at most 40.
            delta_indices=list(range(40)),
            modality_keys=["joint"],
            action_configs=[
                ActionConfig(
                    # Stored targets are absolute q_(t+1); GR00T computes relative actions.
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint",
                )
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }
    if multimodal:
        config.update(
            {
                "tactile": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["rgb"],
                ),
                "pointcloud": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["xyz"],
                ),
            }
        )
    return config
