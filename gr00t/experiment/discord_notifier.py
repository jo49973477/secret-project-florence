# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free Discord webhook client and connectivity test."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import logging
import os
import sys
from typing import Any
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_CONTENT_LENGTH = 2000
_TRUNCATION_MARKER = "\n… message truncated …\n"
_USER_AGENT = "secret-project-florence/1.0"


def validate_discord_configuration(
    *,
    enabled: bool,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the webhook URL from the environment without storing or logging it."""
    if not enabled:
        return None

    env = os.environ if environ is None else environ
    webhook_url = env.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise ValueError("Discord notifications are enabled but DISCORD_WEBHOOK_URL is not set.")
    return webhook_url


def _truncate_content(message: str) -> str:
    """Fit Discord's content limit while retaining both context and trailing details."""
    if len(message) <= _MAX_CONTENT_LENGTH:
        return message
    available = _MAX_CONTENT_LENGTH - len(_TRUNCATION_MARKER)
    prefix_length = available * 2 // 3
    suffix_length = available - prefix_length
    return message[:prefix_length] + _TRUNCATION_MARKER + message[-suffix_length:]


class DiscordNotifier:
    """Small, synchronous, best-effort Discord webhook client."""

    def __init__(
        self,
        *,
        webhook_url: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._opener = urllib.request.urlopen if opener is None else opener

    @classmethod
    def from_environment(
        cls,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> DiscordNotifier:
        webhook_url = validate_discord_configuration(enabled=True, environ=environ)
        return cls(webhook_url=webhook_url, timeout=timeout, opener=opener)

    def send(self, message: str) -> bool:
        """Send one plain-text message, returning false on every transport failure."""
        try:
            payload = json.dumps(
                {"content": _truncate_content(message)}, ensure_ascii=False
            ).encode("utf-8")
            # Never log the request or URL: Discord webhook URLs contain credentials.
            request = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": _USER_AGENT,
                },
                method="POST",
            )
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 204)
                if not 200 <= status < 300:
                    raise RuntimeError("Discord API returned a non-success status")
            return True
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Discord notification failed (HTTP %s); training will continue.",
                exc.code,
            )
            return False
        except Exception as exc:
            # urllib exception strings may contain the credential-bearing request URL.
            logger.warning(
                "Discord notification failed (%s); training will continue.",
                type(exc).__name__,
            )
            return False


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test GR00T Discord notifications.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a test message using DISCORD_WEBHOOK_URL.",
    )
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("the --test flag is required")

    try:
        notifier = DiscordNotifier.from_environment()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    message = "✅ GR00T Discord notification test successful."
    return 0 if notifier.send(message) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
