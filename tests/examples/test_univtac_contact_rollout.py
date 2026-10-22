"""No simulator needed: tactile history and sensor provenance contract."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[2] / "examples/UniVTAC"))
from univtac_rollout.policy import Gr00tUniVTACPolicy

from tests.examples.test_univtac_closed_loop import _modality_configs, _observation


class Client:
    def __init__(self, *a, **kw):
        self.seen = []

    def ping(self):
        return True

    def get_modality_config(self):
        config = _modality_configs()
        config["tactile"] = {"delta_indices": [-1, 0], "modality_keys": ["rgb"]}
        return config

    def reset(self):
        pass

    def close(self):
        pass

    def get_action(self, obs):
        self.seen.append(obs)
        return ({"joint": np.zeros((1, 16, 8), dtype=np.float32)}, {})


def observation(value):
    obs = _observation()
    obs["tactile"] = {"left_gsmini": {"rgb": np.full((8, 8, 3), value, dtype=np.uint8)}}
    return obs


def test_tactile_pair_and_midchunk_history():
    with patch("univtac_rollout.policy.MinimalPolicyClient", Client):
        policy = Gr00tUniVTACPolicy("host", 5555, execution_horizon=2)
        _, formatted = policy.predict_chunk(observation(1), "task")
        assert (formatted["tactile"]["rgb"] == 1).all()
        policy.observe(observation(2))
        policy.observe(observation(3))
        chunk, formatted = policy.predict_chunk(observation(3), "task")
        assert chunk.shape == (2, 8)
        assert (formatted["tactile"]["rgb"][:, 0] == 2).all()
        assert (formatted["tactile"]["rgb"][:, 1] == 3).all()
        policy.reset()
        _, formatted = policy.predict_chunk(observation(4), "task")
        assert (formatted["tactile"]["rgb"] == 4).all()


def test_ambiguous_tactile_source_rejected(monkeypatch):
    with patch("univtac_rollout.policy.MinimalPolicyClient", Client):
        policy = Gr00tUniVTACPolicy("host", 5555)
        obs = observation(1)
        obs["tactile"]["right_gsmini"] = obs["tactile"]["left_gsmini"]
        with pytest.raises(ValueError, match="Select the training sensor"):
            policy.observe(obs)
        monkeypatch.setenv("GR00T_UNIVTAC_TACTILE_KEY", "tactile/left_gsmini/rgb")
        policy.observe(obs)
