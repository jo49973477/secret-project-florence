# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a training command outside its workers and report uncaught process failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys

from gr00t.experiment.discord_notifier import DiscordNotifier


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
    return excerpt or None


def _failure_message(code: int, experiment: str, output: str, log: str | None) -> str:
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
    lines = [
        "🚨 GR00T Training Failed",
        "",
        f"Experiment: {experiment or Path(output).name or 'unknown'}",
        f"Host: {os.uname().nodename}",
        "",
        "Status: FAILED",
        f"Exit code: {exit_code}",
        "Failure source: torchrun supervisor",
        f"Suspected category: {category}",
    ]
    if signal_number is not None:
        lines.append(f"Signal: {signal_number}")
    lines.extend([f"Timestamp: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"])
    if output:
        checkpoints = sorted(Path(output).glob("checkpoint-*"))
        last_checkpoint = checkpoints[-1] if checkpoints else None
        lines.append(f"Output: {output}")
        if last_checkpoint:
            lines.append(f"Last checkpoint: {last_checkpoint}")
    if excerpt:
        lines.extend(["", "Last error / log excerpt:", excerpt])
    if log:
        lines.extend(["", f"Log: {log}"])
    return "\n".join(lines)[:1900]


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
                notifier.send(_failure_message(code, args.experiment, args.output_dir, args.log))
            except Exception:
                # The supervisor must preserve the training process result.
                pass
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
