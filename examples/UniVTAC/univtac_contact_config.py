"""Current/historical tactile RGB plus the unchanged UniVTAC RGB/state contract."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from univtac_config_common import build_univtac_config


config = build_univtac_config(multimodal=True)
del config["pointcloud"]
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
