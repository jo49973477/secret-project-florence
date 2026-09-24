#!/usr/bin/env python3
"""Run one task in one Isaac Sim process for GR00T closed-loop evaluation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from univtac_rollout.core import (
    EpisodeResult,
    InvalidActionError,
    validate_action_step,
    video_output_path,
)
from univtac_rollout.policy import Gr00tUniVTACPolicy


def _parse_args() -> tuple[argparse.Namespace, Any]:
    # AppLauncher must create SimulationApp before envs imports any Omniverse module.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument(
        "--debug-first-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def _load_yaml(path_or_name: str, root: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    path = Path(path_or_name)
    if path.suffix not in (".yml", ".yaml"):
        path = root / "task_config" / f"{path_or_name}.yml"
    elif not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"UniVTAC task config does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"UniVTAC task config must be a mapping: {path}")
    return config, path


def _load_instructions(root: Path, task: str, instruction_type: str) -> list[str]:
    path = root / "instructions" / f"{task}.json"
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    instructions = payload.get(instruction_type)
    if (
        not isinstance(instructions, list)
        or not instructions
        or not all(isinstance(item, str) and item.strip() for item in instructions)
    ):
        raise ValueError(f"{path} has no non-empty {instruction_type!r} instruction list.")
    return instructions


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _runtime_metadata(univtac_root: Path) -> dict[str, Any]:
    import isaaclab

    versions: dict[str, Any] = {
        "python": sys.version,
        "univtac_commit": _git_commit(univtac_root),
        "isaaclab_version": getattr(isaaclab, "__version__", None),
        "isaaclab_module": str(Path(isaaclab.__file__).resolve()),
    }
    for distribution in ("isaacsim", "isaac-sim", "isaaclab"):
        try:
            versions[f"distribution_{distribution}"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    isaaclab_root = next(
        (
            parent
            for parent in Path(isaaclab.__file__).resolve().parents
            if (parent / ".git").is_dir()
        ),
        None,
    )
    versions["isaaclab_commit"] = _git_commit(isaaclab_root) if isaaclab_root else None
    return versions


def _qpos_limits(task: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Read Isaac Lab's articulation soft limits in UniVTAC's 7+1 qpos order."""
    manager = getattr(task, "_robot_manager", None)
    robot = getattr(manager, "robot", None)
    data = getattr(robot, "data", None)
    limits = getattr(data, "soft_joint_pos_limits", None)
    if limits is None:
        return None
    limits = limits.detach().cpu().numpy() if hasattr(limits, "detach") else np.asarray(limits)
    if limits.ndim == 3:
        limits = limits[0]
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError(f"Unexpected articulation limit shape: {limits.shape}")

    arm_ids = np.asarray(manager._arm_ids.detach().cpu().numpy(), dtype=np.int64)
    gripper_ids = np.asarray(manager._gripper_ids.detach().cpu().numpy(), dtype=np.int64)
    if arm_ids.shape != (7,) or gripper_ids.size < 1:
        raise ValueError(
            f"Unexpected UniVTAC DOF mapping: arm_ids={arm_ids}, gripper_ids={gripper_ids}"
        )
    lower = np.concatenate((limits[arm_ids, 0], [np.max(limits[gripper_ids, 0])])).astype(
        np.float32
    )
    upper = np.concatenate((limits[arm_ids, 1], [np.min(limits[gripper_ids, 1])])).astype(
        np.float32
    )
    return lower, upper


def _write_worker_results(
    path: Path, episodes: list[EpisodeResult], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "episodes": [episode.__dict__ for episode in episodes],
                "runtime_metadata": metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finish_video(
    task: Any,
    *,
    output_dir: Path,
    task_name: str,
    seed: int,
    success: bool | None,
    termination_reason: str,
) -> Path:
    tag = "success" if success is True else "failed" if success is False else "error"
    task.clean_cache(result=tag)
    source = task.save_video_path.with_name(f"{task.save_video_path.stem}_{tag}.mp4")
    destination = video_output_path(
        output_dir,
        task=task_name,
        seed=seed,
        success=success,
        termination_reason=termination_reason,
    )
    if not source.is_file():
        raise FileNotFoundError(f"UniVTAC video handler did not create {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite rollout video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), destination)
    return destination


def _run(args: argparse.Namespace) -> None:
    import torch

    root = args.univtac_root.resolve()
    sys.path.insert(0, str(root))
    task_config, task_config_path = _load_yaml(args.task_config, root)
    instructions = _load_instructions(root, args.task, args.instruction_type)
    task_module = importlib.import_module(f"envs.{args.task}")

    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = args.output_dir / "logs" / "univtac" / args.task
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.tactile_sensor_type = task_config.get("sensor_type", env_cfg.tactile_sensor_type)
    env_cfg.scene.num_envs = 1
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device
    if args.step_limit is not None:
        if args.step_limit < 1:
            raise ValueError("--step-limit must be positive.")
        env_cfg.step_lim = args.step_limit
    if env_cfg.video_frequency <= 0:
        raise ValueError("Rollout evaluation requires task_config.video_frequency > 0.")
    required_observations = {
        "camera": {"rgb"},
        "embodiment": {"joint"},
        # UniVTAC's existing get_frame_shot/video path renders both marker views.
        "tactile": {"rgb_marker"},
    }
    for group, required in required_observations.items():
        actual = set(env_cfg.obs_data_type.get(group, []))
        if not required.issubset(actual):
            raise ValueError(
                f"Task config {task_config_path} must request {group} observations {sorted(required)}; "
                f"got {sorted(actual)}."
            )

    metadata = _runtime_metadata(root)
    metadata.update(
        {
            "task_config": str(task_config_path),
            "task": args.task,
            "device": str(env_cfg.sim.device),
        }
    )
    print("Runtime metadata:")
    print(json.dumps(metadata, indent=2))

    task = None
    episodes: list[EpisodeResult] = []
    try:
        policy = Gr00tUniVTACPolicy(
            args.server_host,
            args.server_port,
            execution_horizon=args.execution_horizon,
            timeout_ms=round(args.request_timeout_seconds * 1000),
        )
        try:
            task = task_module.Task(env_cfg, mode="eval")
            limits = _qpos_limits(task)
            if limits is None:
                print(
                    "WARNING: articulation soft joint limits are unavailable; shape/finite checks remain active."
                )
            else:
                print(
                    f"Validated qpos limits lower={limits[0].tolist()} upper={limits[1].tolist()}"
                )

            for episode_index in range(args.episodes):
                seed = args.seed_start + episode_index
                start = time.perf_counter()
                success: bool | None = False
                termination_reason = "step_limit"
                error = ""
                video_path = ""
                instruction = ""
                reset_started = False
                try:
                    task.mode = "eval"
                    task.reset(seed=seed, instructions=instructions)
                    reset_started = True
                    instruction = str(task.instruction)
                    task.mean_steps = task.cfg.step_lim
                    policy.reset()
                    observation = task._get_observations()
                    # Seed every video with the reset state so even a first-action
                    # validation error still yields a useful finalized artifact.
                    task.video_handler.write(task.get_frame_shot(observation))

                    while task.take_action_cnt < task.cfg.step_lim:
                        action_chunk, formatted = policy.predict_chunk(
                            observation, task.instruction
                        )
                        if (
                            args.debug_first_rollout
                            and episode_index == 0
                            and task.take_action_cnt == 0
                        ):
                            debug = {
                                "head": [
                                    formatted["video"]["head"].shape,
                                    str(formatted["video"]["head"].dtype),
                                ],
                                "wrist": [
                                    formatted["video"]["wrist"].shape,
                                    str(formatted["video"]["wrist"].dtype),
                                ],
                                "joint": [
                                    formatted["state"]["joint"].shape,
                                    str(formatted["state"]["joint"].dtype),
                                ],
                                "action_chunk": [action_chunk.shape, str(action_chunk.dtype)],
                            }
                            print(f"First-rollout observation/action contract: {debug}")

                        for predicted in action_chunk:
                            lower, upper = limits if limits is not None else (None, None)
                            action = validate_action_step(predicted, lower=lower, upper=upper)
                            action_tensor = torch.as_tensor(
                                action, dtype=torch.float32, device=task.device
                            )
                            executed, eval_success = task.take_action(
                                action_tensor, action_type="qpos"
                            )
                            if eval_success or task.eval_success:
                                success = True
                                termination_reason = "success"
                                break
                            if not executed or task.check_early_stop():
                                success = False
                                termination_reason = "early_stop"
                                break
                            if task.take_action_cnt >= task.cfg.step_lim:
                                success = False
                                termination_reason = "step_limit"
                                break
                        if (
                            termination_reason != "step_limit"
                            or task.take_action_cnt >= task.cfg.step_lim
                        ):
                            break
                        observation = task._get_observations()
                except InvalidActionError as exc:
                    success = None
                    termination_reason = "invalid_action"
                    error = str(exc)
                    print(f"Seed {seed} invalid action: {error}")
                except Exception:
                    success = None
                    termination_reason = "exception"
                    error = traceback.format_exc()
                    print(f"Seed {seed} exception:\n{error}")
                finally:
                    if reset_started or getattr(task.video_handler, "ffmpeg", None) is not None:
                        try:
                            video_path = str(
                                _finish_video(
                                    task,
                                    output_dir=args.output_dir,
                                    task_name=args.task,
                                    seed=seed,
                                    success=success,
                                    termination_reason=termination_reason,
                                )
                            )
                        except Exception:
                            video_error = traceback.format_exc()
                            error = f"{error}\nVideo finalization error:\n{video_error}".strip()
                            if success is not None:
                                success = None
                                termination_reason = "exception"
                            print(video_error)

                result = EpisodeResult(
                    task=args.task,
                    seed=seed,
                    episode_index=episode_index,
                    success=success,
                    num_steps=int(getattr(task, "take_action_cnt", 0)) if reset_started else 0,
                    sim_steps=int(getattr(task, "step_count", 0)) if reset_started else 0,
                    execution_horizon=args.execution_horizon,
                    checkpoint=args.checkpoint,
                    instruction=instruction,
                    elapsed_seconds=time.perf_counter() - start,
                    termination_reason=termination_reason,
                    video_path=video_path,
                    error=error,
                )
                episodes.append(result)
                _write_worker_results(args.result_json, episodes, metadata)
                print(
                    f"[{episode_index + 1}/{args.episodes}] task={args.task} seed={seed} "
                    f"result={termination_reason} actions={result.num_steps} "
                    f"elapsed={result.elapsed_seconds:.2f}s video={video_path or 'MISSING'}"
                )
        finally:
            policy.close()
    finally:
        if task is not None:
            task.close()


def main() -> None:
    args, simulation_app = _parse_args()
    try:
        _run(args)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
