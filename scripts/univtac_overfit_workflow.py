#!/usr/bin/env python
"""Prepare an independent five-episode subset or evaluate Hydra run checkpoints.

Never replaces an existing subset and never modifies the source dataset.
Training is a separate train_hydra.py command documented in HYDRA_CONFIGURATION.md.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EPISODES = [0, 1, 2, 3, 4]


def prepare(source: Path, target: Path) -> None:
    source, target = source.resolve(), target.resolve()
    if target.exists():
        raise ValueError(
            f"Subset target already exists: {target}; choose a new directory or reuse it"
        )
    if target == source or source in target.parents:
        raise ValueError("Subset must be separate from the source dataset")
    metadata = source / "meta"
    episodes = [
        json.loads(line)
        for line in (metadata / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = {
        int(ep["episode_index"]): ep for ep in episodes if int(ep["episode_index"]) in EPISODES
    }
    if sorted(selected) != EPISODES:
        raise ValueError(f"Source must include all five episode IDs {EPISODES}")
    info = json.loads((metadata / "info.json").read_text())
    target_meta = target / "meta"
    target_meta.mkdir(parents=True)
    for item in metadata.iterdir():
        if item.name in {"info.json", "episodes.jsonl", "stats.json", "relative_stats.json"}:
            continue
        destination = target_meta / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    info.update(
        total_episodes=5,
        total_frames=sum(int(ep["length"]) for ep in selected.values()),
        splits={"train": "0:5"},
        total_chunks=math.ceil(5 / int(info.get("chunks_size", 1000))),
    )
    videos = sum(
        isinstance(value, dict) and value.get("dtype") == "video"
        for value in info.get("features", {}).values()
    )
    if videos:
        info["total_videos"] = 5 * videos
    (target_meta / "info.json").write_text(json.dumps(info, indent=2))
    (target_meta / "episodes.jsonl").write_text(
        "".join(json.dumps(selected[i]) + "\n" for i in EPISODES)
    )
    for directory, dirs, names in os.walk(source, followlinks=True):
        if Path(directory) == source:
            dirs[:] = [name for name in dirs if name != "meta"]
        for name in names:
            item = Path(directory) / name
            match = re.match(r"episode_(\d{6})\.", name)
            if (match and int(match[1]) in EPISODES) or (Path(directory) == source and not match):
                destination = target / item.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(item.resolve())
    parquet = list((target / "data").rglob("episode_*.parquet"))
    if len(parquet) != 5:
        raise ValueError(
            f"Expected five parquet episodes; found {len(parquet)}. Incomplete subset retained at {target} for inspection"
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "gr00t/data/stats.py"),
            "--dataset-path",
            str(target),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(ROOT / "examples/UniVTAC/univtac_config.py"),
        ],
        cwd=ROOT,
        check=True,
    )


def evaluate(dataset: Path, output: Path, gpu: str, horizon: int, steps: int) -> None:
    logs = output / "logs"
    logs.mkdir(exist_ok=True, parents=True)
    rows = ["step,mse,mae\n"]
    checkpoints = sorted(
        (p for p in output.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output}")
    for checkpoint in checkpoints:
        step = int(checkpoint.name.split("-")[1])
        if not (
            any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("pytorch_model*.bin"))
        ):
            raise ValueError(
                f"{checkpoint} has no inference weights. Export native ZeRO state with scripts/export_zero3_checkpoint.py into a separate checkpoint-N directory before same-episode evaluation"
            )
        command = [
            sys.executable,
            str(ROOT / "gr00t/eval/open_loop_eval.py"),
            "--dataset-path",
            str(dataset.resolve()),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--model-path",
            str(checkpoint.resolve()),
            "--traj-ids",
            *map(str, EPISODES),
            "--execution-horizon",
            str(horizon),
            "--steps",
            str(steps),
            "--modality-keys",
            "joint",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (logs / f"eval_{step}.log").write_text(result.stdout)
        print(result.stdout)
        result.check_returncode()
        metrics = [
            re.findall(rf"Average {metric} across all trajs:\s*([^\s]+)", result.stdout)
            for metric in ("MSE", "MAE")
        ]
        if not all(metrics):
            raise ValueError(f"Could not parse metrics for {checkpoint}; see evaluation log")
        rows.append(f"{step},{metrics[0][-1]},{metrics[1][-1]}\n")
    (logs / "memorization_metrics.csv").write_text("".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--target", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--gpu", required=True)
    evaluation.add_argument("--execution-horizon", type=int, default=16)
    evaluation.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.target)
    else:
        evaluate(args.dataset, args.output, args.gpu, args.execution_horizon, args.steps)


if __name__ == "__main__":
    main()
