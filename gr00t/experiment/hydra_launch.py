"""Parent-only preflight, resume validation, snapshots and Torchrun launching."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gr00t.configs.hydra_config import build_gr00t_config, preflight


def validate_checkpoint(path: Path) -> None:
    """Check the same state families as checkpoint_resume.sh without unpickling tensors."""
    if not path.is_dir() or not re.fullmatch(r"checkpoint-\d+", path.name):
        raise ValueError(f"Invalid training checkpoint directory: {path}")
    checks = {
        "trainer state": (path / "trainer_state.json").is_file(),
        "optimizer state": (path / "optimizer.pt").is_file() or any(path.rglob("*optim_states.pt")),
        "scheduler state": (path / "scheduler.pt").is_file() or any(path.rglob("*model_states.pt")),
        "RNG state": any(path.glob("rng_state*.pth")),
        "model weights/partitions": any(path.rglob("*model_states.pt"))
        or any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin")),
    }
    missing = [key for key, present in checks.items() if not present]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is not resumable; missing {', '.join(missing)}. Model-only checkpoints are fine-tuning sources, not full resumes."
        )
    state = json.loads((path / "trainer_state.json").read_text())
    if type(state.get("global_step")) is not int or state["global_step"] != int(
        path.name.split("-")[1]
    ):
        raise ValueError(f"Checkpoint {path}: trainer global_step must match directory step")


def find_latest(output: Path) -> Path:
    candidates = []
    for path in output.glob("checkpoint-*"):
        try:
            validate_checkpoint(path)
        except (ValueError, OSError):
            continue
        candidates.append(path)
    if not candidates:
        raise ValueError(f"No valid resumable checkpoint-* found in {output}")
    return max(candidates, key=lambda path: int(path.name.split("-")[1]))


def resolve_resume(cfg: DictConfig, config) -> Path | None:
    if not cfg.resume.enabled:
        return None
    output = Path(cfg.output_dir).resolve()
    checkpoint = (
        Path(cfg.resume.checkpoint_path).resolve()
        if cfg.resume.checkpoint_path
        else find_latest(output)
    )
    validate_checkpoint(checkpoint)
    if checkpoint.parent != output:
        raise ValueError(
            f"Resume must retain the original output directory; set output_dir={checkpoint.parent}"
        )
    # Hydra provenance is required: legacy checkpoint state alone cannot prove all compatibility.
    saved_path = output / ".hydra" / "config.yaml"
    if not saved_path.is_file():
        raise ValueError(
            f"Cannot verify resume compatibility: missing {saved_path}. Use the legacy launcher for legacy runs, or fine-tune from the model in a fresh run."
        )
    old = OmegaConf.to_container(OmegaConf.load(saved_path), resolve=True)
    new = OmegaConf.to_container(cfg, resolve=True)
    allowed_training_changes = {
        "max_steps",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "use_wandb",
        "wandb_project",
        "telegram_on",
        "telegram_chat_id",
        "telegram_notify_start",
        "telegram_notify_save",
        "telegram_notify_finish",
        "telegram_notify_error",
        "discord_on",
        "discord_notify_start",
        "discord_notify_save",
        "discord_notify_finish",
        "discord_notify_error",
        "enable_training_dashboard",
        "dashboard_refresh_seconds",
    }
    differences = []
    for section in ("model", "data", "training", "hardware"):
        before, after = dict(old[section]), dict(new[section])
        ignored = (
            allowed_training_changes
            if section == "training"
            else {"master_port", "cuda_visible_devices", "dataloader_num_workers"}
            if section == "hardware"
            else set()
        )
        for key in sorted(set(before) | set(after)):
            if key not in ignored and before.get(key) != after.get(key):
                differences.append(f"{section}.{key}")
    if differences:
        raise ValueError(
            f"Incompatible resume changes: {', '.join(differences)}. Optimizer/scheduler state restores saved LRs; LR, modality, architecture and optimizer structure changes require a fresh fine-tune."
        )
    if old["name"] != new["name"]:
        raise ValueError("Resume must use the original experiment name")
    cfg.run_id = old["run_id"]
    step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
    if (
        config.training.max_steps <= step
        or config.training.max_steps < old["training"]["max_steps"]
    ):
        raise ValueError(
            "Resume max_steps must exceed the saved step and must not shorten the original duration"
        )
    config.training.resume_from_checkpoint = str(checkpoint)
    # Preserve the legacy construction path from the original base model. Trainer
    # restores the checkpoint (including native ZeRO partitions) after construction.
    return checkpoint


def write_snapshots(cfg: DictConfig, config, checkpoint: Path | None = None) -> Path:
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    # Keep initial provenance immutable. Resumes get their own small manifest directory.
    snapshot = output / ".hydra"
    if checkpoint:
        snapshot = snapshot / f"resume_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    snapshot.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, snapshot / "config.yaml", resolve=True)
    hydra = HydraConfig.get() if HydraConfig.initialized() else OmegaConf.create({})
    OmegaConf.save(hydra, snapshot / "hydra.yaml", resolve=False)
    overrides = list(hydra.overrides.task) if HydraConfig.initialized() else []
    OmegaConf.save(OmegaConf.create(overrides), snapshot / "overrides.yaml")
    config.save(snapshot / "gr00t.yaml")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    metadata = {
        "git_commit": commit,
        "git_dirty": dirty,
        "experiment_name": cfg.name,
        "run_id": output.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration_groups": OmegaConf.to_container(hydra.runtime.choices)
        if HydraConfig.initialized()
        else {},
        "overrides": overrides,
        "base_model": cfg.model.base_model_path,
        "datasets": [asdict(ds) for ds in config.data.datasets],
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "precision": "bf16" if config.training.bf16 else "fp16" if config.training.fp16 else "fp32",
        "resume_source": str(checkpoint) if checkpoint else None,
        **preflight(cfg, config),
    }
    (snapshot / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return snapshot / "config.yaml"


def worker_command(cfg: DictConfig, resolved_path: Path) -> list[str]:
    worker = Path(__file__).resolve().parents[2] / "scripts" / "train_hydra_worker.py"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={cfg.hardware.num_gpus}",
        f"--master-port={cfg.hardware.master_port}",
        str(worker),
        str(resolved_path),
    ]


def validate_dataset_inputs(cfg: DictConfig) -> None:
    """Check local dataset prerequisites and an explicitly asserted independent subset."""
    expected = cfg.data.get("expected_episode_ids")
    for dataset in cfg.data.datasets:
        for directory in dataset.dataset_paths:
            path = Path(directory)
            if not path.is_dir():
                raise ValueError(
                    f"Dataset directory does not exist: {path}. Set data.datasets or FLORENCE_DATASET to an existing converted dataset"
                )
            for name in ("info.json", "modality.json"):
                if not (path / "meta" / name).is_file():
                    raise ValueError(f"Dataset is missing meta/{name}: {path}")
            if expected is not None:
                episodes = [
                    json.loads(line)
                    for line in (path / "meta/episodes.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                actual = sorted(int(ep["episode_index"]) for ep in episodes)
                if actual != sorted(expected):
                    raise ValueError(
                        f"Dataset {path} must be an independent subset containing exactly episode IDs {list(expected)}, got {actual}. Use univtac_overfit_workflow.py prepare"
                    )


def launch(cfg: DictConfig) -> int:
    config = build_gr00t_config(cfg)
    # Dry-run does not even read checkpoint metadata; resume validation is a real-launch operation.
    if cfg.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
        print(json.dumps(preflight(cfg, config), indent=2), flush=True)
        return 0
    if "RANK" in os.environ:
        raise ValueError("Run train_hydra.py once as the parent; it launches Torchrun itself")
    validate_dataset_inputs(cfg)
    checkpoint = resolve_resume(cfg, config)
    output = Path(cfg.output_dir).resolve()
    if not checkpoint and output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Fresh run requires an empty output_dir: {output}; use resume.enabled=true or a new run_id"
        )
    # Serialize resolved literals so ranks cannot resolve environment/time independently.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.output_dir = str(output)
    config.training.output_dir = str(output)
    if checkpoint:
        cfg.resume.checkpoint_path = str(checkpoint)
        cfg.resume.auto_find_latest = False
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print(json.dumps(preflight(cfg, config), indent=2), flush=True)
    path = write_snapshots(cfg, config, checkpoint)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cfg.hardware.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env.setdefault("LOGURU_LEVEL", "INFO")
    env["GR00T_HYDRA_CONFIG"] = str(path)
    env.update(
        GR00T_EXPERIMENT_NAME=cfg.name,
        GR00T_OUTPUT_DIR=str(output),
        GR00T_TRAIN_LOG=str(output / "logs/train.log"),
    )
    command = worker_command(cfg, path)
    # Reuse the notification supervisor for abrupt failures outside Trainer callbacks.
    command = [
        sys.executable,
        "-m",
        "gr00t.experiment.torchrun_supervisor",
        "--marker",
        str(output / ".gr00t-failure-notification-delivered"),
        "--",
        *command,
    ]
    if config.training.discord_on:
        env["GR00T_SUPERVISOR_NOTIFY"] = "1"
    # A separate process group allows forwarding termination to supervisor, Torchrun and ranks.
    with (output / "logs/train.log").open("a", buffering=1) as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def forward(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    return code if code >= 0 else 128 - code
