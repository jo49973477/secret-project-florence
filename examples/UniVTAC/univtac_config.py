# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UniVTAC RGB baseline modality configuration."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from univtac_config_common import build_univtac_config


univtac_config = build_univtac_config(multimodal=False)

register_modality_config(univtac_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
