"""Small CLI/configuration bridge; trainctl owns all run lifecycle decisions."""

from contextlib import contextmanager
from importlib.util import find_spec
import logging
from pathlib import Path
import sys
import tempfile
import threading
import tomllib


logger = logging.getLogger(__name__)
WEBHOOK_VARIABLES = ("TRAINCTL_DISCORD_WEBHOOK_URL", "DISCORD_WEBHOOK_URL", "DISCORD_WEBHOOK")


def discord_environment(env, settings):
    """Resolve aliases without logging, serializing, or probing a credential."""
    env = dict(env)
    configured = [(key, env[key].strip()) for key in WEBHOOK_VARIABLES if env.get(key, "").strip()]
    if configured:
        env["TRAINCTL_DISCORD_WEBHOOK_URL"] = configured[0][1]
        if len({value for _, value in configured}) > 1:
            logger.warning(
                "Conflicting Discord webhook variables; precedence is "
                "TRAINCTL_DISCORD_WEBHOOK_URL > DISCORD_WEBHOOK_URL > DISCORD_WEBHOOK. "
                "Using %s; values withheld.",
                configured[0][0],
            )
    discord = settings.get("notifications", {}).get("discord", {}) or settings.get("discord", {})
    if discord.get("enabled") is False:
        logger.warning(
            "Discord disabled by explicit trainctl TOML enabled=false; "
            "remove that setting or set enabled=true to receive notifications."
        )
        return env, "disabled (explicit trainctl TOML enabled=false)"
    webhook = env.get("TRAINCTL_DISCORD_WEBHOOK_URL") or discord.get("webhook_url")
    if not webhook:
        logger.warning(
            "Discord webhook missing; set TRAINCTL_DISCORD_WEBHOOK_URL in the launching shell."
        )
        return env, "missing"
    if find_spec("httpx") is None:
        logger.warning(
            "Discord unavailable: install trainctl[discord]==0.2.0 in this Python environment (httpx missing)."
        )
        return env, "unavailable (httpx missing)"
    try:
        # The exported public adapter validates offline; this performs no HTTP request.
        from trainctl.integrations.discord import DiscordWebhookAdapter

        DiscordWebhookAdapter(webhook)
    except (ImportError, ValueError, TypeError):
        logger.warning(
            "Discord webhook invalid or transport unavailable; configure a valid HTTPS Discord webhook and install trainctl[discord]==0.2.0."
        )
        return env, "invalid"
    return env, "configured (delivery checked by trainctl at lifecycle events)"


def check_trainctl():
    if find_spec("trainctl") is None:
        raise RuntimeError(
            "trainctl is enabled but unavailable in this Python environment. "
            "Install WAH's locked dependencies with uv sync --locked, or set trainctl.enabled=false."
        )
    try:
        import tomli_w  # noqa: F401
        from trainctl import tracker

        if not all(
            hasattr(tracker, name) for name in ("init", "define_metrics", "log", "checkpoint")
        ):
            raise ImportError("incompatible tracker")
    except ImportError as exc:
        raise RuntimeError(
            "Install WAH's pinned trainctl 0.2.0 dependency with uv sync --locked"
        ) from exc


def trainctl_command(cfg, output, command):
    return [
        sys.executable,
        "-m",
        "gr00t.experiment.trainctl_cli",
        "run",
        "--name",
        f"{cfg.name}-{cfg.run_id}",
        "--checkpoint-dir",
        str(Path(output).resolve()),
        "--total-steps",
        str(cfg.training.max_steps),
        "--message-mode",
        cfg.trainctl.message_mode,
        "--",
        *command,
    ]


@contextmanager
def trainctl_run(cfg, output, command, env):
    env = dict(env)
    if not cfg.trainctl.enabled:
        # Explicit debugging opt-out also prevents attaching to an outer SDK endpoint.
        for key in ("TRAINCTL_RUN_ID", "TRAINCTL_SOCKET", "GR00T_TRAINCTL_ACTIVE"):
            env.pop(key, None)
        yield command, env
        return
    env["GR00T_TRAINCTL_ACTIVE"] = "1"
    env["GR00T_TRAINCTL_CHECKPOINTS"] = "1" if cfg.trainctl.notify_checkpoints else "0"
    # An explicitly outer-supervised process must never acquire another supervisor.
    if env.get("TRAINCTL_RUN_ID"):
        raise ValueError(
            "Hydra already runs inside trainctl; launch Hydra directly to avoid nested supervision"
        )
    import tomli_w

    source = Path(env.get("TRAINCTL_CONFIG", "~/.config/trainctl/config.toml")).expanduser()
    settings = tomllib.loads(source.read_text()) if source.exists() else {}
    env, discord_status = discord_environment(env, settings)
    interactive = sys.stdout.isatty() and cfg.training.get("enable_training_dashboard") is not False
    print(
        "trainctl: enabled\n"
        f"message mode: {cfg.trainctl.message_mode}\n"
        f"Discord: {discord_status}\n"
        f"Dashboard: {'Rich interactive (parent)' if interactive else 'text fallback'}\n"
        f"Checkpoint notifications: {'enabled' if cfg.trainctl.notify_checkpoints else 'disabled'}",
        flush=True,
    )
    # Polling observes directories before HF has finished writing ZeRO partitions.
    # The SDK is the sole checkpoint source; preserve all other user TOML settings.
    settings.setdefault("checkpoints", {})["pattern"] = "(?!)"
    with tempfile.TemporaryDirectory(prefix="wah-trainctl-") as directory:
        path = Path(directory) / "config.toml"
        path.write_text(tomli_w.dumps(settings))
        path.chmod(0o600)
        env["TRAINCTL_CONFIG"] = str(path)
        # A distinct log directory per invocation also handles resumed runs.
        log_dir = Path(output) / "logs" / "trainctl" / Path(directory).name
        env["TRAINCTL_LOG_DIR"] = str(log_dir.resolve())
        yield trainctl_command(cfg, output, command), env


class LogRelay:
    """Stream trainctl's captured worker log back to the existing console/train.log.

    The pinned public CLI doesn't echo worker output. Read only its documented
    log files; never query its private database or supervisor implementation.
    """

    def __init__(self, directory, log):
        self.directory = Path(directory)
        self.log = log
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        handle = None
        try:
            while True:
                stopping = self.stop.is_set()
                if handle is None:
                    paths = list(self.directory.glob("*.log"))
                    if paths:
                        handle = paths[0].open(encoding="utf-8", errors="replace")
                if handle is not None:
                    text = handle.read()
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        self.log.write(text)
                if stopping:
                    break
                self.stop.wait(0.05)
        finally:
            if handle is not None:
                handle.close()

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
