# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, dependency-free process-memory diagnostics for model loading."""

import logging
import os


def log_model_load_memory(label: str) -> None:
    """Log Linux RSS/high-water memory when GROOT_DEBUG_MODEL_LOAD_MEMORY is enabled."""
    if os.environ.get("GROOT_DEBUG_MODEL_LOAD_MEMORY", "").lower() not in {"1", "true", "yes"}:
        return
    values = {}
    try:
        with open("/proc/self/status") as status:
            for line in status:
                key, separator, value = line.partition(":")
                if separator and key in {"VmRSS", "VmHWM"}:
                    values[key] = value.strip()
    except OSError:
        values = {}
    logging.info(
        "Model-load memory [%s] rank=%s pid=%d RSS=%s peak=%s",
        label,
        os.environ.get("RANK", "0"),
        os.getpid(),
        values.get("VmRSS", "unavailable"),
        values.get("VmHWM", "unavailable"),
    )
