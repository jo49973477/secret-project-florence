# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys

import msgpack
import numpy as np
import pytest


UNIVTAC_EXAMPLE_ROOT = Path(__file__).parents[2] / "examples" / "UniVTAC"
sys.path.insert(0, str(UNIVTAC_EXAMPLE_ROOT))

from univtac_rollout.client import CompatibleMsgSerializer  # noqa: E402
from univtac_rollout.core import (  # noqa: E402
    CANONICAL_TASKS,
    EpisodeResult,
    InvalidActionError,
    aggregate_results,
    classify_result,
    extract_action_chunk,
    format_univtac_observation,
    resolve_tasks,
    validate_action_step,
    video_output_path,
    write_results,
)
from univtac_rollout.policy import validate_rgb_checkpoint_contract  # noqa: E402


def _observation() -> dict:
    return {
        "observation": {
            "head": {"rgb": np.zeros((270, 480, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((270, 480, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.arange(9, dtype=np.float64)},
    }


def _episode(
    task: str,
    index: int,
    success: bool | None,
    reason: str,
    *,
    steps: int = 10,
) -> EpisodeResult:
    return EpisodeResult(
        task=task,
        seed=1_000_000 + index,
        episode_index=index,
        success=success,
        num_steps=steps,
        execution_horizon=1,
        checkpoint="checkpoint-2000",
        instruction="do the task",
        elapsed_seconds=1.5,
        termination_reason=reason,
        video_path=f"videos/{task}-{index}.mp4",
    )


def _modality_configs(action_horizon: int = 16) -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": ["head", "wrist"]},
        "state": {"delta_indices": [0], "modality_keys": ["joint"]},
        "action": {
            "delta_indices": list(range(action_horizon)),
            "modality_keys": ["joint"],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }


def test_observation_format_matches_rgb_checkpoint_contract() -> None:
    formatted = format_univtac_observation(_observation(), "lift the can")
    assert formatted["video"]["head"].shape == (1, 1, 270, 480, 3)
    assert formatted["video"]["wrist"].shape == (1, 1, 270, 480, 3)
    assert formatted["video"]["head"].dtype == np.uint8
    assert formatted["state"]["joint"].shape == (1, 1, 8)
    assert formatted["state"]["joint"].dtype == np.float32
    np.testing.assert_array_equal(formatted["state"]["joint"][0, 0], np.arange(8, dtype=np.float32))
    assert formatted["language"] == {"annotation.human.task_description": [["lift the can"]]}


def test_observation_format_rejects_wrong_rgb_dtype_and_short_joint_state() -> None:
    observation = _observation()
    observation["observation"]["head"]["rgb"] = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        format_univtac_observation(observation, "task")

    observation = _observation()
    observation["embodiment"]["joint"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 8"):
        format_univtac_observation(observation, "task")


def test_action_extraction_accepts_one_batch_and_rejects_malformed_actions() -> None:
    chunk = np.arange(4 * 8, dtype=np.float32).reshape(1, 4, 8)
    extracted = extract_action_chunk([{"joint": chunk}, {}])
    assert extracted.shape == (4, 8)
    assert extracted.dtype == np.float32

    with pytest.raises(InvalidActionError, match="8 dimensions"):
        extract_action_chunk([{"joint": np.zeros((1, 4, 7), dtype=np.float32)}, {}])
    chunk[0, 2, 3] = np.nan
    with pytest.raises(InvalidActionError, match="NaN/Inf"):
        extract_action_chunk([{"joint": chunk}, {}])


def test_action_limit_validation_rejects_without_clipping() -> None:
    action = np.zeros(8, dtype=np.float32)
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    np.testing.assert_array_equal(validate_action_step(action, lower=lower, upper=upper), action)
    action[6] = 1.1
    with pytest.raises(InvalidActionError, match="dof 6"):
        validate_action_step(action, lower=lower, upper=upper)
    assert action[6] == pytest.approx(1.1)


def test_checkpoint_contract_requires_rgb_schema_and_sufficient_horizon() -> None:
    validate_rgb_checkpoint_contract(_modality_configs(), execution_horizon=16)
    with pytest.raises(ValueError, match="exceeds checkpoint action horizon"):
        validate_rgb_checkpoint_contract(_modality_configs(4), execution_horizon=8)
    multimodal = _modality_configs()
    multimodal["tactile"] = {"delta_indices": [0], "modality_keys": ["rgb"]}
    with pytest.raises(ValueError, match="RGB rollout adapter"):
        validate_rgb_checkpoint_contract(multimodal, execution_horizon=1)


def test_minimal_serializer_round_trips_requests_and_decodes_msgpack_numpy() -> None:
    request = {"observation": {"joint": np.arange(8, dtype=np.float32)}}
    decoded = CompatibleMsgSerializer.from_bytes(CompatibleMsgSerializer.to_bytes(request))
    np.testing.assert_array_equal(decoded["observation"]["joint"], request["observation"]["joint"])

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    wire = msgpack.packb(
        {
            b"nd": True,
            b"type": array.dtype.str,
            b"kind": array.dtype.kind,
            b"shape": array.shape,
            b"data": array.tobytes(),
        },
        use_bin_type=True,
    )
    np.testing.assert_array_equal(CompatibleMsgSerializer.from_bytes(wire), array)


def test_task_resolution_and_result_classification() -> None:
    assert resolve_tasks(["all"]) == list(CANONICAL_TASKS)
    assert resolve_tasks(["lift_can", "insert_HDMI"]) == ["lift_can", "insert_HDMI"]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_tasks(["lift_can", "lift_can"])
    assert classify_result(True, "success") == "success"
    assert classify_result(False, "early_stop") == "failure"
    assert classify_result(None, "invalid_action") == "error"


def test_video_output_path_separates_success_failure_and_error(tmp_path: Path) -> None:
    success = video_output_path(
        tmp_path,
        task="lift_can",
        seed=7,
        success=True,
        termination_reason="success",
    )
    failure = video_output_path(
        tmp_path,
        task="lift_can",
        seed=8,
        success=False,
        termination_reason="step_limit",
    )
    error = video_output_path(
        tmp_path,
        task="lift_can",
        seed=9,
        success=None,
        termination_reason="exception",
    )
    assert success == tmp_path / "videos/success/lift_can/lift_can_seed_7_success.mp4"
    assert failure == tmp_path / "videos/failure/lift_can/lift_can_seed_8_failure.mp4"
    assert error == tmp_path / "videos/error/lift_can/lift_can_seed_9_error.mp4"


def test_aggregation_uses_macro_average_and_excludes_errors() -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success"),
        _episode("lift_bottle", 1, False, "step_limit"),
        _episode("lift_can", 0, True, "success"),
        _episode("lift_can", 1, True, "success"),
        _episode("lift_can", 2, True, "success"),
        _episode("lift_can", 3, None, "exception"),
    ]
    aggregate = aggregate_results(episodes, ["lift_bottle", "lift_can"])
    assert aggregate["macro_average_percent"] == pytest.approx(75.0)
    assert aggregate["micro_average_percent"] == pytest.approx(80.0)
    assert aggregate["total_errors"] == 1
    assert aggregate["episode_counts_differ"] is True


def test_macro_average_is_not_reported_when_a_task_has_no_valid_episode() -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success"),
        _episode("lift_can", 0, None, "invalid_action"),
    ]
    aggregate = aggregate_results(episodes, ["lift_bottle", "lift_can"])
    assert aggregate["macro_average"] is None
    assert aggregate["total_errors"] == 1


def test_result_files_contain_episode_rows_and_macro_summary(tmp_path: Path) -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success", steps=4),
        _episode("lift_bottle", 1, False, "early_stop", steps=8),
    ]
    aggregate = write_results(
        tmp_path,
        episodes,
        ["lift_bottle"],
        checkpoint="/models/checkpoint-2000",
        execution_horizon=1,
        episodes_per_task=2,
    )
    assert aggregate["macro_average_percent"] == pytest.approx(50.0)
    for filename in ("episodes.csv", "summary.csv", "summary.json", "summary.md", "summary.txt"):
        assert (tmp_path / filename).is_file()
    text = (tmp_path / "summary.txt").read_text()
    assert "Lift Bottle" in text
    assert "Average" in text
    assert "50.00%" in text
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["summary"]["tasks"][0]["successes"] == 1
    assert payload["episodes"][1]["termination_reason"] == "early_stop"
