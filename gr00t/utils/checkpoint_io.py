"""Local torch checkpoint reads without eager tensor-storage copies."""

import os
from pathlib import Path
import zipfile

import torch


def load_torch_checkpoint(path, *, weights_only: bool, memory_map: bool = True):
    """Use PyTorch's existing mmap for ZIP checkpoints, retaining legacy formats.

    mmap maps file-backed storages privately. No clone, dtype conversion, cache
    rewrite, or full-state broadcast is performed. Preserve weights_only policy
    at the caller; in particular, never retry an unsafe unpickle automatically.
    """
    path = Path(path)
    use_mmap = memory_map and os.environ.get("GROOT_MODEL_LOAD_MMAP", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    kwargs = {"mmap": True} if use_mmap and zipfile.is_zipfile(path) else {}
    return torch.load(path, map_location="cpu", weights_only=weights_only, **kwargs)
