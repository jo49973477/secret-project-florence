# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a training command outside its workers and report uncaught process failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from gr00t.experiment.discord_notifier import DiscordNotifier, format_training_event


def _safe_log_excerpt(path: str | None) -> str | None:
    if not path:
        return None
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    except OSError:
        return None
    excerpt = "\n".join(lines)[-1200:]
    excerpt = re.sub(
        r"https://(?:discord(?:app)?\.com)/api/webhooks/\S+", "[webhook redacted]", excerpt
    )
    excerpt = re.sub(r"(?i)(token|password|secret)(\s*[=:]\s*)\S+", r"\1\2[redacted]", excerpt)
    excerpt = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", excerpt)
    return excerpt or None


def _last_checkpoint(output: str) -> Path | None:
    if not output:
        return None
    candidates = []
    for path in Path(output).glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.removeprefix("checkpoint-")), path))
        except ValueError:
            continue
    return max(candidates)[1] if candidates else None


def _checkpoint_step(checkpoint: Path | None) -> int | None:
    if checkpoint is None:
        return None
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        step = int(state["global_step"])
        return step if step >= 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _exposure_fields(output: str, step: int | None) -> list[tuple[str, object]]:
    if not output or step is None or (Path(output) / "sample_exposure_unavailable").exists():
        return []
    try:
        manifest = json.loads((Path(output) / "sample_exposure.json").read_text(encoding="utf-8"))
        batch = int(manifest["per_device_train_batch_size"]) * int(manifest["world_size"])
        batch *= int(manifest["gradient_accumulation_steps"])
        total = int(manifest["total_training_samples"])
        if min(batch, total) <= 0:
            return []
    except (OSError, ValueError, KeyError, TypeError):
        return []
    samples = step * batch
    return [
        ("Samples Seen (at Last Checkpoint)", f"{samples:,}"),
        ("Effective Epoch (at Last Checkpoint)", f"{samples / total:.4f}"),
    ]


def _failure_message(
    code: int, experiment: str, output: str, log: str | None, duration: float | None = None
) -> str:
    if code < 0:
        signal_number = -code
        category = (
            "Worker terminated by signal (SIGKILL may be OOM Killer or external kill)"
            if signal_number == 9
            else f"Worker terminated by signal {signal_number}"
        )
        exit_code = 128 + signal_number
    else:
        signal_number = None
        exit_code = code
        category = (
            "torchrun ChildFailedError / distributed worker failure"
            if code == 1
            else "Unknown nonzero process exit"
        )
        if code == 130 or code == 143:
            category = "Explicit cancellation / termination signal"
    excerpt = _safe_log_excerpt(log)
    checkpoint = _last_checkpoint(output)
    step = _checkpoint_step(checkpoint)
    fields = [
        ("Experiment", experiment or Path(output).name or "unknown"),
        ("Host", os.uname().nodename),
        ("Last Checkpoint Global Step", step),
        *_exposure_fields(output, step),
        ("Last Checkpoint", checkpoint),
        ("Exception Type", "WorkerSignal" if signal_number is not None else "WorkerExit"),
        ("Error Message", category),
        ("Exit Code", exit_code),
        ("Failure Source", "torchrun supervisor"),
        ("Signal", signal_number),
        ("Training Duration", f"{duration:.0f}s" if duration is not None else None),
        ("Timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Log Excerpt", excerpt),
        ("Log Location", log),
    ]
    return format_training_event("error", fields)[:1900]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--experiment", default=os.environ.get("GR00T_EXPERIMENT_NAME", ""))
    parser.add_argument("--output-dir", default=os.environ.get("GR00T_OUTPUT_DIR", ""))
    parser.add_argument("--log", default=os.environ.get("GR00T_TRAIN_LOG", ""))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    env = dict(os.environ, GR00T_FAILURE_NOTIFICATION_MARKER=str(marker))
    started = time.monotonic()
    try:
        result = subprocess.run(command, env=env, check=False)
        code = result.returncode
    except KeyboardInterrupt:
        code = 130
    if (
        code
        and os.environ.get("GR00T_SUPERVISOR_NOTIFY") == "1"
        and os.environ.get("DISCORD_WEBHOOK_URL")
    ):
        if not marker.exists():
            try:
                notifier = DiscordNotifier.from_environment(timeout=3.0)
                notifier.send(
                    _failure_message(
                        code, args.experiment, args.output_dir, args.log, time.monotonic() - started
                    )
                )
            except Exception:
                # The supervisor must preserve the training process result.
                pass
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
