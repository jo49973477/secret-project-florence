# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated non-Hydra Discord transport; Hydra notifications use trainctl."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import logging
import os
import sys
import time
from typing import Any
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_CONTENT_LENGTH = 2000
_TRUNCATION_MARKER = "\n… message truncated …\n"
_USER_AGENT = "secret-project-florence/1.0"
_MAX_ATTEMPTS = 3
_MAX_BACKOFF_SECONDS = 1.0

KOYORI_START = (
    "콘코요~! 조수군, 새로운 모델 학습 실험 스타트야! holoX의 두뇌인 코요리가 옆에서 지켜보고 "
    "있으니까 이번 Loss는 무조건 완벽하게 수렴할 거라구~! 🧪✨"
)
KOYORI_FINISH = (
    "조수군, 실험 대성공이야~! 마지막 에폭까지 가중치가 아주 예쁘게 농축됐어! 역시 코요리의 "
    "완벽한 조력 덕분이지? 얼른 결과 보러 와~! 🐾💖"
)
KOYORI_ERROR = (
    "와아아앗 조수군 큰일이야!! 실험실 플라스크가… 아니, 그래디언트가 폭발해서 학습이 멈췄어!! "
    "코요리는 아무것도 안 건드렸다구, 빨리 와서 로그 좀 봐줘어어~! 💥😭"
)


def format_training_event(event: str, fields: list[tuple[str, object]]) -> str:
    """Format Discord-only training content; the transport enforces 2000 characters."""
    heading, intro = {
        "start": ("🧪 Training Started", KOYORI_START),
        "finish": ("✅ Training Finished", KOYORI_FINISH),
        "error": ("🚨 Training Failed", KOYORI_ERROR),
        "save": ("💾 GR00T Checkpoint Saved", None),
    }[event]
    lines = [heading, ""]
    if intro is not None:
        lines.extend([intro, ""])
    lines.extend(f"{label}: {value}" for label, value in fields if value is not None)
    return "\n".join(lines)


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
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._opener = urllib.request.urlopen if opener is None else opener
        self._sleep = sleep

    @classmethod
    def from_environment(
        cls,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> DiscordNotifier:
        webhook_url = validate_discord_configuration(enabled=True, environ=environ)
        return cls(webhook_url=webhook_url, timeout=timeout, opener=opener, sleep=sleep)

    def send(self, message: str) -> bool:
        """Send one plain-text message, returning false on every transport failure."""
        payload = json.dumps({"content": _truncate_content(message)}, ensure_ascii=False).encode(
            "utf-8"
        )
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
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                with self._opener(request, timeout=self._timeout) as response:
                    status = getattr(response, "status", 204)
                if 200 <= status < 300:
                    return True
                retry_after = None
                retryable = status in {429, 500, 502, 503, 504}
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                retryable = status in {429, 500, 502, 503, 504}
            except Exception as exc:
                status = None
                retry_after = None
                retryable = isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))
                error_type = type(exc).__name__

            if retryable and attempt < _MAX_ATTEMPTS:
                logger.warning("Discord notification retry %s/%s", attempt + 1, _MAX_ATTEMPTS)
                try:
                    delay = (
                        float(retry_after)
                        if retry_after is not None
                        else 0.25 * (2 ** (attempt - 1))
                    )
                except (TypeError, ValueError):
                    delay = 0.25 * (2 ** (attempt - 1))
                self._sleep(min(_MAX_BACKOFF_SECONDS, max(0.0, delay)))
                continue
            if status is not None:
                logger.warning("Discord notification failed: HTTP %s", status)
            else:
                logger.warning("Discord notification failed (%s)", error_type)
            logger.warning("Discord notification delivery failed permanently")
            return False
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
