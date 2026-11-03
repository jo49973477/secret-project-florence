import importlib.util
from pathlib import Path
import sys


CONFIG_DIR = Path(__file__).parents[3] / "examples" / "UniVTAC"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CONFIG_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_univtac_rgb_and_multimodal_share_common_modalities(monkeypatch):
    monkeypatch.syspath_prepend(str(CONFIG_DIR))
    rgb = _load("univtac_config").univtac_config
    multimodal = _load("univtac_multimodal_config").univtac_multimodal_config
    for modality in ("video", "state", "action", "language"):
        assert rgb[modality] == multimodal[modality]
    assert len(rgb["action"].delta_indices) == 16
    assert "pointcloud" in multimodal and "tactile" in multimodal
    del sys.modules["univtac_config_common"]


def test_univtac_horizon_can_be_explicitly_set_to_40(monkeypatch):
    monkeypatch.syspath_prepend(str(CONFIG_DIR))
    from univtac_config_common import build_univtac_config

    action_config = build_univtac_config(multimodal=False, action_horizon=40)["action"]
    assert len(action_config.delta_indices) == 40
    del sys.modules["univtac_config_common"]
