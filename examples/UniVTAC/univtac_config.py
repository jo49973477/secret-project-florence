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

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


univtac_config = {
    # ============================================================
    # 1. RGB observation
    # ============================================================
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "wrist"],
    ),
    # ============================================================
    # 2. Robot proprioceptive state
    # ============================================================
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint"],
    ),
    # ============================================================
    # 3. Robot action
    # ============================================================
    "action": ModalityConfig(
        # UniVTAC's ACT baseline uses 50, but the N1.7 base model predicts at
        # most 40 actions and rejects longer modality horizons at startup.
        delta_indices=list(range(16)),
        modality_keys=["joint"],
        action_configs=[
            ActionConfig(
                # Dataset stores q_{t+1}, not q_{t+1} - q_t.
                # GR00T will convert absolute q into relative action internally.
                rep=ActionRepresentation.RELATIVE,
                # Joint-space control
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                # Compute relative action against current joint state
                state_key="joint",
            ),
        ],
    ),
    # ============================================================
    # 4. Language instruction
    # ============================================================
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.task_description",
        ],
    ),
}


register_modality_config(
    univtac_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
