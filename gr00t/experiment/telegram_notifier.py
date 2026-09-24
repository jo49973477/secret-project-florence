# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free Telegram Bot API client and connectivity test."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import logging
import os
import sys
from typing import Any
import urllib.parse
import urllib.request


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5.0


def validate_telegram_configuration(
    *,
    enabled: bool,
    chat_id: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str] | None:
    """Resolve Telegram credentials without storing them in training configuration.

    The explicit chat ID takes priority over ``TELEGRAM_CHAT_ID``. The bot token is
    intentionally available only through ``TELEGRAM_BOT_TOKEN``.
    """
    if not enabled:
        return None

    env = os.environ if environ is None else environ
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("Telegram notifications are enabled but TELEGRAM_BOT_TOKEN is not set.")

    resolved_chat_id = (chat_id or env.get("TELEGRAM_CHAT_ID", "")).strip()
    if not resolved_chat_id:
        raise ValueError(
            "Telegram notifications are enabled but no chat ID was provided. "
            "Set TELEGRAM_CHAT_ID or pass --telegram-chat-id."
        )
    return token, resolved_chat_id


class TelegramNotifier:
    """Small, synchronous, best-effort Telegram Bot API client."""

    def __init__(
        self,
        *,
        token: str = "",
        chat_id: str = "",
        enabled: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._enabled = enabled
        self._timeout = timeout
        self._opener = urllib.request.urlopen if opener is None else opener

    @classmethod
    def from_environment(
        cls,
        *,
        chat_id: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> TelegramNotifier:
        token, resolved_chat_id = validate_telegram_configuration(
            enabled=True,
            chat_id=chat_id,
            environ=environ,
        )
        return cls(
            token=token,
            chat_id=resolved_chat_id,
            timeout=timeout,
            opener=opener,
        )

    def send(self, message: str) -> bool:
        """Send one plain-text message, returning false on every transport failure."""
        if not self._enabled:
            return False

        try:
            payload = urllib.parse.urlencode({"chat_id": self._chat_id, "text": message}).encode(
                "utf-8"
            )
            # Never log this URL: it contains the bot token.
            request = urllib.request.Request(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise RuntimeError("Telegram API returned a non-success status")
            return True
        except Exception as exc:
            # Exception strings from HTTP clients can contain the request URL (and token),
            # so log only the exception type.
            logger.warning(
                "Telegram notification failed (%s); training will continue.",
                type(exc).__name__,
            )
            return False


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test GR00T Telegram notifications.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a test message using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
    )
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("the --test flag is required")

    try:
        notifier = TelegramNotifier.from_environment()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if notifier.send("✅ GR00T Telegram notification test successful.") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
