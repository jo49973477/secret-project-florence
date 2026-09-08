# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).parents[2] / "examples" / "UniVTAC" / "evaluate_baseline_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_baseline_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)
aggregate_seed_metrics = DIAGNOSTICS.aggregate_seed_metrics
apply_rgb_condition = DIAGNOSTICS.apply_rgb_condition
cyclic_derangement_indices = DIAGNOSTICS.cyclic_derangement_indices
persistence_metrics_for_arrays = DIAGNOSTICS.persistence_metrics_for_arrays
write_csv = DIAGNOSTICS._write_csv


def _trajectory() -> pd.DataFrame:
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(5)]
    return pd.DataFrame(
        {
            "video.head": frames,
            "video.wrist": [frame + 10 for frame in frames],
            "state.joint": [np.array([value], dtype=np.float32) for value in range(5)],
            "action.joint": [np.array([value + 1], dtype=np.float32) for value in range(5)],
            "language.annotation.human.task_description": ["task"] * 5,
        }
    )


def test_black_affects_all_rgb_only_and_preserves_shape_dtype() -> None:
    trajectory = _trajectory()
    black = apply_rgb_condition(
        trajectory,
        video_keys=["head", "wrist"],
        condition="black",
        seed=3,
        trajectory_id=0,
    )
    for key in ("video.head", "video.wrist"):
        for original, transformed in zip(trajectory[key], black[key]):
            assert transformed.shape == original.shape
            assert transformed.dtype == original.dtype
            assert np.count_nonzero(transformed) == 0
    pd.testing.assert_series_equal(black["state.joint"], trajectory["state.joint"])
    pd.testing.assert_series_equal(black["action.joint"], trajectory["action.joint"])
    pd.testing.assert_series_equal(
        black["language.annotation.human.task_description"],
        trajectory["language.annotation.human.task_description"],
    )


def test_shuffle_is_a_seeded_permutation_with_no_fixed_indices() -> None:
    trajectory = _trajectory()
    first = apply_rgb_condition(
        trajectory,
        video_keys=["head", "wrist"],
        condition="shuffled",
        seed=7,
        trajectory_id=2,
    )
    repeated = apply_rgb_condition(
        trajectory,
        video_keys=["head", "wrist"],
        condition="shuffled",
        seed=7,
        trajectory_id=2,
    )
    for key in ("video.head", "video.wrist"):
        original_values = [int(frame[0, 0, 0]) for frame in trajectory[key]]
        shuffled_values = [int(frame[0, 0, 0]) for frame in first[key]]
        repeated_values = [int(frame[0, 0, 0]) for frame in repeated[key]]
        assert sorted(shuffled_values) == sorted(original_values)
        assert shuffled_values == repeated_values
        assert all(left != right for left, right in zip(original_values, shuffled_values))
    assert not np.array_equal(
        cyclic_derangement_indices(101, seed=7, trajectory_id=2, camera_key="head"),
        cyclic_derangement_indices(101, seed=8, trajectory_id=2, camera_key="head"),
    )


def test_shuffle_rejects_single_frame_trajectory() -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        cyclic_derangement_indices(1, seed=0, trajectory_id=0, camera_key="head")


def test_persistence_metrics_known_values() -> None:
    state = np.arange(4, dtype=np.float32)[:, None]
    action = state + 1
    metrics = persistence_metrics_for_arrays(state, action, execution_horizon=2, steps=4)
    assert metrics["persistence_one_step"] == pytest.approx((1.0, 1.0, 4))
    assert metrics["persistence_chunk_hold"] == pytest.approx((2.5, 1.5, 4))


def test_persistence_fails_on_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="compatible"):
        persistence_metrics_for_arrays(
            np.zeros((3, 2)), np.zeros((3, 3)), execution_horizon=2, steps=3
        )


def test_metric_aggregation_uses_population_std_and_normal_delta() -> None:
    raw = pd.DataFrame(
        [
            {
                "checkpoint": "checkpoint-5",
                "checkpoint_step": 5,
                "condition": "normal",
                "seed": 0,
                "mse": 1.0,
                "mae": 2.0,
            },
            {
                "checkpoint": "checkpoint-5",
                "checkpoint_step": 5,
                "condition": "normal",
                "seed": 1,
                "mse": 3.0,
                "mae": 4.0,
            },
            {
                "checkpoint": "checkpoint-5",
                "checkpoint_step": 5,
                "condition": "black",
                "seed": 0,
                "mse": 4.0,
                "mae": 5.0,
            },
            {
                "checkpoint": "checkpoint-5",
                "checkpoint_step": 5,
                "condition": "black",
                "seed": 1,
                "mse": 6.0,
                "mae": 7.0,
            },
        ]
    )
    summary = aggregate_seed_metrics(raw).set_index("condition")
    assert summary.loc["normal", "mse_mean"] == pytest.approx(2.0)
    assert summary.loc["normal", "mse_std"] == pytest.approx(1.0)
    assert summary.loc["normal", "delta_mse_vs_normal"] == pytest.approx(0.0)
    assert summary.loc["black", "delta_mse_vs_normal"] == pytest.approx(3.0)
    assert summary.loc["black", "percent_mse_degradation_vs_normal"] == pytest.approx(150.0)


def test_aggregated_metrics_csv_round_trip(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "checkpoint": "checkpoint-5",
                "checkpoint_step": 5,
                "condition": "normal",
                "seed": 0,
                "mse": 1.0,
                "mae": 2.0,
            }
        ]
    )
    output = tmp_path / "summary_metrics.csv"
    write_csv(aggregate_seed_metrics(raw), output)
    reloaded = pd.read_csv(output)
    assert reloaded.loc[0, "mse_mean"] == pytest.approx(1.0)
    assert reloaded.loc[0, "mse_std"] == pytest.approx(0.0)
    assert "percent_mae_degradation_vs_normal" in reloaded.columns
