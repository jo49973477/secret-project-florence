#!/usr/bin/env python3
"""Orchestrate closed-loop GR00T rollouts across UniVTAC tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from univtac_rollout.core import EpisodeResult, resolve_tasks, write_results
from univtac_rollout.policy import Gr00tUniVTACPolicy


MODE_EPISODES = {"smoke": 5, "quick": 20, "full": 100}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--mode", choices=tuple(MODE_EPISODES), default="smoke")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--task-config", default="clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/univtac_rollout_eval"))
    parser.add_argument(
        "--python-command",
        default=sys.executable,
        help="Isaac Python command, e.g. '/path/to/IsaacLab/isaaclab.sh -p'.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug-first-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _validate(args: argparse.Namespace) -> tuple[list[str], int]:
    tasks = resolve_tasks(args.tasks)
    episodes = (
        args.episodes_per_task if args.episodes_per_task is not None else MODE_EPISODES[args.mode]
    )
    if episodes < 1:
        raise ValueError("--episodes-per-task must be positive.")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be nonnegative.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be positive.")
    root = args.univtac_root.resolve()
    required = [root / "envs" / "_base_task.py", root / "task_config", root / "instructions"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"--univtac-root is not a UniVTAC checkout; missing: {missing}")
    worker = Path(__file__).with_name("run_closed_loop_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(f"Rollout worker is missing: {worker}")
    return tasks, episodes


def _load_worker_result(path: Path) -> tuple[list[EpisodeResult], dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return (
        [EpisodeResult(**row) for row in payload["episodes"]],
        payload.get("runtime_metadata", {}),
    )


def _stream_process(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + shlex.join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks, episodes_per_task = _validate(args)
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "logs" / "worker_results"
    existing = [
        path for path in (output_dir / "episodes.csv", output_dir / "summary.json") if path.exists()
    ]
    existing.extend(path for path in result_dir.glob("*.json") if path.exists())
    if existing:
        raise FileExistsError(
            "Refusing to mix a new evaluation with existing results. Choose a new "
            f"--output-dir. Existing paths: {[str(path) for path in existing]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking GR00T server {args.server_host}:{args.server_port} and checkpoint contract...")
    with Gr00tUniVTACPolicy(
        args.server_host,
        args.server_port,
        execution_horizon=args.execution_horizon,
        timeout_ms=round(args.request_timeout_seconds * 1000),
    ):
        pass

    python_command = shlex.split(args.python_command)
    if not python_command:
        raise ValueError("--python-command cannot be empty.")
    worker = Path(__file__).with_name("run_closed_loop_worker.py").resolve()
    all_episodes: list[EpisodeResult] = []
    runtime_metadata: dict[str, Any] = {}

    for task_index, task in enumerate(tasks):
        result_json = result_dir / f"{task}.json"
        command = [
            *python_command,
            str(worker),
            "--univtac-root",
            str(args.univtac_root.resolve()),
            "--task",
            task,
            "--task-config",
            args.task_config,
            "--instruction-type",
            args.instruction_type,
            "--episodes",
            str(episodes_per_task),
            "--seed-start",
            str(args.seed_start),
            "--execution-horizon",
            str(args.execution_horizon),
            "--checkpoint",
            args.checkpoint,
            "--server-host",
            args.server_host,
            "--server-port",
            str(args.server_port),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--output-dir",
            str(output_dir),
            "--result-json",
            str(result_json),
            "--device",
            args.device,
        ]
        if args.headless:
            command.append("--headless")
        command.append(
            "--debug-first-rollout"
            if args.debug_first_rollout and task_index == 0
            else "--no-debug-first-rollout"
        )
        if args.step_limit is not None:
            command.extend(("--step-limit", str(args.step_limit)))

        print(f"\nStarting UniVTAC task {task} ({task_index + 1}/{len(tasks)})")
        return_code = _stream_process(
            command,
            cwd=args.univtac_root.resolve(),
            log_path=output_dir / "logs" / f"{task}.log",
        )
        if result_json.is_file():
            task_episodes, task_metadata = _load_worker_result(result_json)
            all_episodes.extend(task_episodes)
            runtime_metadata[task] = task_metadata
            write_results(
                output_dir,
                all_episodes,
                tasks,
                checkpoint=args.checkpoint,
                execution_horizon=args.execution_horizon,
                episodes_per_task=episodes_per_task,
                metadata={
                    "mode": args.mode,
                    "seed_start": args.seed_start,
                    "server": f"{args.server_host}:{args.server_port}",
                    "runtime_by_task": runtime_metadata,
                },
            )
        if return_code != 0:
            raise RuntimeError(
                f"UniVTAC worker for {task} exited with code {return_code}. "
                f"See {output_dir / 'logs' / f'{task}.log'}"
            )

    aggregate = write_results(
        output_dir,
        all_episodes,
        tasks,
        checkpoint=args.checkpoint,
        execution_horizon=args.execution_horizon,
        episodes_per_task=episodes_per_task,
        metadata={
            "mode": args.mode,
            "seed_start": args.seed_start,
            "server": f"{args.server_host}:{args.server_port}",
            "runtime_by_task": runtime_metadata,
        },
    )
    print("\n" + (output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"Results: {output_dir}")
    return aggregate


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
