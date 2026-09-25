# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible Telegram callback imports.

Training event formatting is provider-neutral; this module remains so existing imports of
``TelegramTrainingCallback`` and ``TrainingNotificationContext`` continue to work.
"""

from gr00t.experiment.training_notification_callback import (
    TrainingNotificationCallback,
    TrainingNotificationContext,
)


__all__ = ["TelegramTrainingCallback", "TrainingNotificationContext"]


class TelegramTrainingCallback(TrainingNotificationCallback):
    """Backward-compatible name for the provider-neutral training callback."""
