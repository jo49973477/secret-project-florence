from types import SimpleNamespace

import numpy as np
import torch

from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


def _stats(shape):
    size = int(np.prod(shape))
    values = np.arange(size, dtype=np.float32).reshape(shape)
    return {name: values.copy() for name in ("min", "q01")} | {
        name: values + 2 for name in ("max", "q99")
    } | {"mean": values.copy(), "std": np.ones(shape, dtype=np.float32)}


def test_horizon_dependent_action_statistics_keep_feature_dimension_and_round_trip():
    horizon, action_dim = 5, 3
    statistics = {
        "robot": {
            "state": {"joint": _stats((action_dim,))},
            "action": {"joint": _stats((action_dim,))},
            "relative_action": {"joint": _stats((horizon, action_dim))},
        }
    }
    config = {
        "robot": {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
            "action": ModalityConfig(
                delta_indices=list(range(horizon)),
                modality_keys=["joint"],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.RELATIVE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                        state_key="joint",
                    )
                ],
            ),
        }
    }
    processor = StateActionProcessor(
        config, statistics=statistics, use_relative_action=True, clip_outliers=False
    )

    params = processor.norm_params["robot"]["action"]["joint"]
    assert params["min"].shape == (horizon, action_dim)
    assert processor.get_action_dim("robot") == action_dim
    values = params["min"] + (params["max"] - params["min"]) * 0.37
    state = {"joint": np.zeros((1, action_dim), dtype=np.float32)}
    normalized = processor.apply_action({"joint": values.copy()}, "robot", state=state)
    restored = processor.unapply_action(normalized, "robot", state=state)
    np.testing.assert_allclose(normalized["joint"], np.full_like(values, -0.26), atol=1e-6)
    np.testing.assert_allclose(restored["joint"], values, atol=1e-6)


def test_absolute_action_statistics_keep_vector_dimension():
    stats = _stats((7,))
    processor = StateActionProcessor(
        {
            "robot": {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
                "action": ModalityConfig(delta_indices=[0], modality_keys=["joint"]),
            }
        },
        statistics={"robot": {"state": {"joint": stats}, "action": {"joint": stats}}},
    )
    assert processor.norm_params["robot"]["action"]["joint"]["dim"].item() == 7
    assert processor.get_action_dim("robot") == 7


def test_inference_action_mask_uses_feature_dimension_not_statistics_horizon():
    processor = object.__new__(Gr00tN1d7Processor)
    processor.modality_configs = {
        "robot": {"action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["joint"])}
    }
    processor.max_action_horizon = 40
    processor.max_action_dim = 32
    processor.state_action_processor = SimpleNamespace(get_action_dim=lambda _tag: 7)
    mask = processor._make_action_mask("robot")
    assert mask.shape == (40, 32)
    assert torch.all(mask[:16, :7] == 1)
    assert torch.count_nonzero(mask[16:]) == 0
    assert torch.count_nonzero(mask[:, 7:]) == 0
