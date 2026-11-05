"""Path-independent identity for an inference checkpoint (standard library only)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def checkpoint_identity(path: str | Path) -> str:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is unavailable locally: {root}")
    weights = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin"}
    )
    if not weights:
        raise ValueError(
            f"{root} has no inference weights. Export a DeepSpeed ZeRO-3 checkpoint with "
            "scripts/export_zero3_checkpoint.py before serving or evaluating it."
        )
    files = weights + [p for p in (root / "config.json",) if p.is_file()]
    if not (root / "config.json").is_file():
        raise ValueError(f"Checkpoint is missing config.json: {root}")
    processor = root / "processor"
    for candidate in (root / "processor_config.json", processor / "processor_config.json"):
        if candidate.is_file():
            files.append(candidate)
            break
    else:
        raise ValueError(f"Checkpoint is missing processor_config.json: {root}")
    if processor.is_dir():
        files.extend(p for p in processor.rglob("*") if p.is_file())
    for pattern in (
        "statistics.json",
        "embodiment_id.json",
        "preprocessor_config.json",
        "tokenizer*",
        "special_tokens*",
        "vocab*",
        "merges.txt",
        "chat_template*",
    ):
        files.extend(p for p in root.glob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for file in sorted(set(files), key=lambda p: str(p.relative_to(root))):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()
