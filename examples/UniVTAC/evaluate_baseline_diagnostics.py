#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline RGB-ablation and persistence diagnostics for the UniVTAC baseline.

The evaluator intentionally loads one checkpoint at a time and evaluates every
requested RGB condition and seed before releasing it. Metrics use decoded,
unnormalized action coordinates, matching ``gr00t/eval/open_loop_eval.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
import gc
import hashlib
import importlib.util
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("univtac_diagnostics")
RGB_CONDITIONS = ("normal", "black", "shuffled")


def _load_univtac_config() -> dict[str, Any]:
    """Load the adjacent config without relying on ``examples`` being a package."""
    config_path = Path(__file__).with_name("univtac_config.py")
    spec = importlib.util.spec_from_file_location("univtac_diagnostic_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load UniVTAC modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.univtac_config


def cyclic_derangement_indices(
    length: int,
    *,
    seed: int,
    trajectory_id: int,
    camera_key: str,
) -> np.ndarray:
    """Return a deterministic nonzero cyclic permutation for one camera stream."""
    if length < 2:
        raise ValueError(
            "The shuffled RGB condition requires at least two frames per trajectory; "
            f"trajectory {trajectory_id}, camera {camera_key!r} has {length}."
        )
    camera_token = int.from_bytes(
        hashlib.blake2s(camera_key.encode("utf-8"), digest_size=4).digest(), "little"
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(trajectory_id), camera_token])
    )
    offset = int(rng.integers(1, length))
    return (np.arange(length, dtype=np.int64) + offset) % length


def apply_rgb_condition(
    trajectory: pd.DataFrame,
    *,
    video_keys: Sequence[str],
    condition: str,
    seed: int,
    trajectory_id: int,
) -> pd.DataFrame:
    """Apply an RGB condition without changing non-video trajectory columns."""
    if condition not in RGB_CONDITIONS:
        raise ValueError(f"Unknown RGB condition {condition!r}; expected one of {RGB_CONDITIONS}.")
    if not video_keys:
        raise ValueError("The policy declares no RGB views; UniVTAC RGB diagnostics cannot run.")

    transformed = trajectory.copy(deep=False)
    for key in video_keys:
        column = f"video.{key}"
        if column not in trajectory.columns:
            raise KeyError(
                f"Required RGB column {column!r} is absent. Available columns: "
                f"{list(trajectory.columns)}"
            )
        original_frames = trajectory[column].tolist()
        if condition == "normal":
            continue
        if condition == "black":
            transformed[column] = [np.zeros_like(np.asarray(frame)) for frame in original_frames]
            continue
        indices = cyclic_derangement_indices(
            len(original_frames), seed=seed, trajectory_id=trajectory_id, camera_key=key
        )
        transformed[column] = [original_frames[index] for index in indices]
    return transformed


def error_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Compute scalar MSE and MAE after strict shape/finite validation."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(
            f"Metric shape mismatch: target={target.shape}, prediction={prediction.shape}"
        )
    if target.size == 0:
        raise ValueError("Cannot compute metrics for empty arrays.")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs contain NaN or Inf.")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(error**2)), float(np.mean(np.abs(error)))


def persistence_metrics_for_arrays(
    state: np.ndarray,
    action: np.ndarray,
    *,
    execution_horizon: int,
    steps: int,
) -> dict[str, tuple[float, float, int]]:
    """Calculate one-step and chunk-hold persistence metrics for one trajectory."""
    state = np.asarray(state)
    action = np.asarray(action)
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError(
            f"Persistence expects 2D [time, dim] arrays: {state.shape}, {action.shape}"
        )
    if state.shape != action.shape:
        raise ValueError(
            "Persistence requires compatible q_t state and q_(t+1) action coordinates; "
            f"got state={state.shape}, action={action.shape}."
        )
    if execution_horizon < 1:
        raise ValueError("execution_horizon must be positive.")
    actual_steps = min(int(steps), len(state))
    if actual_steps < 1:
        raise ValueError("No valid persistence steps are available.")

    one_step_mse, one_step_mae = error_metrics(action[:actual_steps], state[:actual_steps])
    held_predictions = []
    held_targets = []
    for step in range(0, actual_steps, execution_horizon):
        count = min(execution_horizon, actual_steps - step)
        held_predictions.append(np.repeat(state[step : step + 1], count, axis=0))
        held_targets.append(action[step : step + count])
    chunk_mse, chunk_mae = error_metrics(
        np.concatenate(held_targets, axis=0), np.concatenate(held_predictions, axis=0)
    )
    return {
        "persistence_one_step": (one_step_mse, one_step_mae, actual_steps),
        "persistence_chunk_hold": (chunk_mse, chunk_mae, actual_steps),
    }


def aggregate_seed_metrics(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per seed and add degradation relative to normal RGB."""
    required = {"checkpoint", "checkpoint_step", "condition", "seed", "mse", "mae"}
    missing = required.difference(raw_metrics.columns)
    if missing:
        raise ValueError(f"Cannot aggregate metrics; missing columns: {sorted(missing)}")
    grouped = (
        raw_metrics.groupby(["checkpoint", "checkpoint_step", "condition"], sort=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", lambda values: float(np.std(values, ddof=0))),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda values: float(np.std(values, ddof=0))),
        )
        .reset_index()
    )
    normal = grouped[grouped["condition"] == "normal"][
        ["checkpoint", "checkpoint_step", "mse_mean", "mae_mean"]
    ].rename(columns={"mse_mean": "normal_mse", "mae_mean": "normal_mae"})
    if len(normal) != raw_metrics[["checkpoint", "checkpoint_step"]].drop_duplicates().shape[0]:
        raise ValueError("Every checkpoint requires a normal condition for degradation metrics.")
    grouped = grouped.merge(normal, on=["checkpoint", "checkpoint_step"], validate="many_to_one")
    grouped["delta_mse_vs_normal"] = grouped["mse_mean"] - grouped["normal_mse"]
    grouped["delta_mae_vs_normal"] = grouped["mae_mean"] - grouped["normal_mae"]
    grouped["percent_mse_degradation_vs_normal"] = np.where(
        grouped["normal_mse"] != 0,
        100.0 * grouped["delta_mse_vs_normal"] / grouped["normal_mse"],
        np.nan,
    )
    grouped["percent_mae_degradation_vs_normal"] = np.where(
        grouped["normal_mae"] != 0,
        100.0 * grouped["delta_mae_vs_normal"] / grouped["normal_mae"],
        np.nan,
    )
    return grouped.drop(columns=["normal_mse", "normal_mae"])


def _stack_columns(trajectory: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    arrays = []
    for column in columns:
        if column not in trajectory.columns:
            raise KeyError(f"Required column {column!r} is missing from the trajectory.")
        arrays.append(np.vstack([np.asarray(value) for value in trajectory[column]]))
    return np.concatenate(arrays, axis=-1)


def _checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", str(checkpoint))
    if match is None:
        raise ValueError(
            f"Cannot infer checkpoint step from {checkpoint}; expected a checkpoint-<step> path."
        )
    return int(match.group(1))


def _validate_univtac_persistence_contract(modality_configs: dict[str, Any]) -> None:
    state = modality_configs["state"]
    action = modality_configs["action"]
    if state.delta_indices != [0]:
        raise ValueError(
            f"UniVTAC persistence expects state.delta_indices=[0], got {state.delta_indices}"
        )
    if state.modality_keys != action.modality_keys:
        raise ValueError(
            "UniVTAC persistence requires identical state/action coordinate keys; "
            f"got state={state.modality_keys}, action={action.modality_keys}."
        )
    if action.action_configs is None:
        raise ValueError(
            "UniVTAC action_configs are required to verify state/action correspondence."
        )
    for key, config in zip(action.modality_keys, action.action_configs):
        if config.state_key != key:
            raise ValueError(
                f"Action key {key!r} is relative to state key {config.state_key!r}; "
                "the persistence baseline cannot compare them directly."
            )


def _run_persistence(
    loader: Any,
    *,
    trajectory_ids: Sequence[int],
    steps: int,
    execution_horizon: int,
) -> pd.DataFrame:
    _validate_univtac_persistence_contract(loader.modality_configs)
    state_columns = [f"state.{key}" for key in loader.modality_configs["state"].modality_keys]
    action_columns = [f"action.{key}" for key in loader.modality_configs["action"].modality_keys]
    by_baseline: dict[str, list[tuple[float, float, int]]] = {
        "persistence_one_step": [],
        "persistence_chunk_hold": [],
    }
    for trajectory_id in trajectory_ids:
        trajectory = loader[trajectory_id]
        state = _stack_columns(trajectory, state_columns)
        action = _stack_columns(trajectory, action_columns)
        metrics = persistence_metrics_for_arrays(
            state, action, execution_horizon=execution_horizon, steps=steps
        )
        for baseline, values in metrics.items():
            by_baseline[baseline].append(values)
    rows = []
    for baseline, values in by_baseline.items():
        rows.append(
            {
                "baseline": baseline,
                "execution_horizon": 1 if baseline == "persistence_one_step" else execution_horizon,
                "trajectory_count": len(values),
                "evaluated_steps": sum(value[2] for value in values),
                "mse": float(np.mean([value[0] for value in values])),
                "mae": float(np.mean([value[1] for value in values])),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_loaded_trajectory(
    policy: Any,
    trajectory: pd.DataFrame,
    *,
    modality_configs: dict[str, Any],
    embodiment_tag: Any,
    steps: int,
    execution_horizon: int,
) -> tuple[float, float, int]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    PolicyHorizonSpec.from_modality_config(modality_configs, n_action_steps=execution_horizon)
    actual_steps = min(steps, len(trajectory))
    observation_configs = deepcopy(modality_configs)
    observation_configs.pop("action")
    action_keys = modality_configs["action"].modality_keys
    predictions = []
    for step in range(0, actual_steps, execution_horizon):
        point = extract_step_data(
            trajectory, step, observation_configs, embodiment_tag=embodiment_tag
        )
        observation: dict[str, Any] = {}
        for key, value in point.states.items():
            observation[f"state.{key}"] = value
        for key, value in point.images.items():
            observation[f"video.{key}"] = np.asarray(value)
        for language_key in modality_configs["language"].modality_keys:
            observation[language_key] = point.text
        parsed = parse_observation_gr00t(observation, modality_configs)
        action_chunk, _ = policy.get_action(parsed)
        flat_action = parse_action_gr00t(action_chunk)
        for chunk_index in range(execution_horizon):
            predictions.append(
                np.concatenate(
                    [
                        np.atleast_1d(flat_action[f"action.{key}"][chunk_index])
                        for key in action_keys
                    ],
                    axis=0,
                )
            )
    target = _stack_columns(trajectory, [f"action.{key}" for key in action_keys])[:actual_steps]
    prediction = np.asarray(predictions)[:actual_steps]
    mse, mae = error_metrics(target, prediction)
    return mse, mae, actual_steps


def _evaluate_combination(
    policy: Any,
    loader: Any,
    *,
    trajectory_ids: Sequence[int],
    condition: str,
    seed: int,
    embodiment_tag: Any,
    steps: int,
    execution_horizon: int,
) -> tuple[float, float, int]:
    from gr00t.utils.determinism import seed_everything

    trajectory_metrics = []
    total_steps = 0
    video_keys = loader.modality_configs["video"].modality_keys
    for trajectory_id in trajectory_ids:
        LOGGER.info("Evaluating trajectory=%d condition=%s seed=%d", trajectory_id, condition, seed)
        trajectory = loader[trajectory_id]
        conditioned = apply_rgb_condition(
            trajectory,
            video_keys=video_keys,
            condition=condition,
            seed=seed,
            trajectory_id=trajectory_id,
        )
        # Reset before each trajectory so all RGB conditions receive identical
        # flow-noise streams for a given user seed and trajectory.
        seed_everything(seed, deterministic_algorithms=False)
        policy.reset()
        mse, mae, evaluated_steps = _evaluate_loaded_trajectory(
            policy,
            conditioned,
            modality_configs=loader.modality_configs,
            embodiment_tag=embodiment_tag,
            steps=steps,
            execution_horizon=execution_horizon,
        )
        trajectory_metrics.append((mse, mae))
        total_steps += evaluated_steps
    return (
        float(np.mean([value[0] for value in trajectory_metrics])),
        float(np.mean([value[1] for value in trajectory_metrics])),
        total_steps,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.execution_horizon < 1 or args.denoising_steps < 1:
        raise ValueError("steps, execution-horizon, and denoising-steps must all be positive.")
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset_path}")
    if not args.checkpoints:
        raise ValueError("At least one --checkpoints path is required.")
    for checkpoint in args.checkpoints:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        _checkpoint_step(checkpoint)
    if len(set(args.trajectory_ids)) != len(args.trajectory_ids):
        raise ValueError("--trajectory-ids contains duplicates.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be unique nonnegative integers.")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("--conditions contains duplicates.")
    unknown = set(args.conditions).difference(RGB_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown --conditions values: {sorted(unknown)}")
    if "normal" not in args.conditions:
        raise ValueError("--conditions must include normal so degradation can be calculated.")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    _validate_args(args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    univtac_config = _load_univtac_config()

    persistence_config = {
        "state": deepcopy(univtac_config["state"]),
        "action": deepcopy(univtac_config["action"]),
    }
    persistence_loader = LeRobotEpisodeLoader(args.dataset_path, persistence_config)
    invalid_ids = [value for value in args.trajectory_ids if value >= len(persistence_loader)]
    if invalid_ids or any(value < 0 for value in args.trajectory_ids):
        raise IndexError(
            f"Trajectory IDs {invalid_ids or args.trajectory_ids} are invalid for a dataset with "
            f"{len(persistence_loader)} trajectories."
        )
    LOGGER.info("Computing persistence baselines before loading checkpoints")
    persistence = _run_persistence(
        persistence_loader,
        trajectory_ids=args.trajectory_ids,
        steps=args.steps,
        execution_horizon=args.execution_horizon,
    )
    _write_csv(persistence, args.result_dir / "persistence_metrics.csv")
    del persistence_loader

    raw_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(args.checkpoints):
        checkpoint_step = _checkpoint_step(checkpoint)
        LOGGER.info("Loading checkpoint once: %s", checkpoint)
        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(checkpoint),
            device=args.device,
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        modality_configs = policy.get_modality_config()
        _validate_univtac_persistence_contract(modality_configs)
        video_config = modality_configs.get("video")
        if video_config is None or not video_config.modality_keys:
            raise ValueError(f"Checkpoint {checkpoint} does not declare RGB modalities.")
        video_keys = video_config.modality_keys
        expected_video_keys = univtac_config["video"].modality_keys
        if video_keys != expected_video_keys:
            raise ValueError(
                f"Checkpoint {checkpoint} RGB views {video_keys} do not match the UniVTAC "
                f"baseline views {expected_video_keys}."
            )
        from gr00t.eval._horizon_contract import PolicyHorizonSpec

        PolicyHorizonSpec.from_modality_config(
            modality_configs, n_action_steps=args.execution_horizon
        )
        LOGGER.info("Affecting all policy RGB views: %s", video_keys)
        loader = LeRobotEpisodeLoader(args.dataset_path, modality_configs)

        first_result: tuple[float, float, int] | None = None
        for condition in args.conditions:
            for seed in args.seeds:
                LOGGER.info(
                    "checkpoint_step=%d condition=%s seed=%d", checkpoint_step, condition, seed
                )
                mse, mae, evaluated_steps = _evaluate_combination(
                    policy,
                    loader,
                    trajectory_ids=args.trajectory_ids,
                    condition=condition,
                    seed=seed,
                    embodiment_tag=embodiment_tag,
                    steps=args.steps,
                    execution_horizon=args.execution_horizon,
                )
                row = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": checkpoint_step,
                    "condition": condition,
                    "seed": seed,
                    "mse": mse,
                    "mae": mae,
                    "trajectory_count": len(args.trajectory_ids),
                    "evaluated_steps": evaluated_steps,
                }
                raw_rows.append(row)
                _write_csv(pd.DataFrame(raw_rows), args.result_dir / "raw_metrics.csv")
                LOGGER.info("mse=%.10f mae=%.10f", mse, mae)
                if condition == args.conditions[0] and seed == args.seeds[0]:
                    first_result = (mse, mae, evaluated_steps)

        if args.verify_reproducibility and checkpoint_index == 0:
            assert first_result is not None
            condition = args.conditions[0]
            seed = args.seeds[0]
            LOGGER.info(
                "Repeating checkpoint_step=%d condition=%s seed=%d for reproducibility",
                checkpoint_step,
                condition,
                seed,
            )
            repeated = _evaluate_combination(
                policy,
                loader,
                trajectory_ids=args.trajectory_ids,
                condition=condition,
                seed=seed,
                embodiment_tag=embodiment_tag,
                steps=args.steps,
                execution_horizon=args.execution_horizon,
            )
            exact = all(first == second for first, second in zip(first_result[:2], repeated[:2]))
            reproduction_rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": checkpoint_step,
                    "condition": condition,
                    "seed": seed,
                    "first_mse": first_result[0],
                    "repeated_mse": repeated[0],
                    "mse_abs_diff": abs(first_result[0] - repeated[0]),
                    "first_mae": first_result[1],
                    "repeated_mae": repeated[1],
                    "mae_abs_diff": abs(first_result[1] - repeated[1]),
                    "exact_match": exact,
                    "close_match": bool(
                        np.allclose(first_result[:2], repeated[:2], rtol=1e-7, atol=1e-10)
                    ),
                }
            )
            _write_csv(
                pd.DataFrame(reproduction_rows),
                args.result_dir / "reproducibility_check.csv",
            )
            LOGGER.info(
                "Reproducibility exact_match=%s close_match=%s",
                exact,
                reproduction_rows[-1]["close_match"],
            )

        del loader, policy
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    raw = pd.DataFrame(raw_rows)
    summary = aggregate_seed_metrics(raw)
    _write_csv(raw, args.result_dir / "raw_metrics.csv")
    _write_csv(summary, args.result_dir / "summary_metrics.csv")

    print("\nGR00T summary (unnormalized action coordinates):")
    print(
        summary[
            ["checkpoint_step", "condition", "mse_mean", "mse_std", "mae_mean", "mae_std"]
        ].to_string(index=False)
    )
    print("\nPersistence:")
    print(persistence[["baseline", "mse", "mae"]].to_string(index=False))
    print(f"\nResults: {args.result_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--trajectory-ids", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", choices=RGB_CONDITIONS, required=True)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Logical torch device. The server script maps the selected physical GPU to cuda:0.",
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Repeat the first checkpoint/condition/seed combination and compare metrics.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
