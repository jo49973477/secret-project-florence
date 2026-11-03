#!/usr/bin/env python
"""Small fresh-process eager/mmap comparison; never accesses the HF cache or GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument("--worker", choices=["eager", "mmap"])
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.worker:
        from gr00t.utils.checkpoint_io import load_torch_checkpoint
        from gr00t.utils.model_load_diagnostics import model_load_snapshot
        import torch

        torch.set_num_threads(1)
        before = model_load_snapshot()
        start = time.monotonic()
        state = load_torch_checkpoint(
            args.checkpoint, weights_only=True, memory_map=args.worker == "mmap"
        )
        after_load = model_load_snapshot()
        load_seconds = time.monotonic() - start
        checksum = state["weight"].sum().item()  # Touch all pages, not only the header.
        after_touch = model_load_snapshot()
        print(
            json.dumps(
                {
                    "mode": args.worker,
                    "file_bytes": Path(args.checkpoint).stat().st_size,
                    "before": before,
                    "after_load": after_load,
                    "after_touch": after_touch,
                    "load_seconds": load_seconds,
                    "total_seconds": time.monotonic() - start,
                    "checksum": checksum,
                }
            )
        )
        return
    if not 1 <= args.size_mib <= 256:
        parser.error("--size-mib must be 1..256 for this lightweight benchmark")
    import torch

    with tempfile.TemporaryDirectory(prefix="gr00t-checkpoint-io-") as directory:
        path = Path(directory) / "synthetic.pth"
        torch.save({"weight": torch.ones(args.size_mib * 1024 * 1024 // 4)}, path)
        results = []
        for mode in ("eager", "mmap"):
            output = subprocess.check_output(
                [sys.executable, __file__, "--worker", mode, "--checkpoint", str(path)],
                text=True,
                timeout=60,
                env={**os.environ, "GROOT_MODEL_LOAD_MMAP": "1"},
            )
            results.append(json.loads(output.splitlines()[-1]))
        assert results[0]["checksum"] == results[1]["checksum"]
        print(
            json.dumps(
                {"cache": "warm page cache; no drop_caches performed", "results": results}, indent=2
            )
        )


if __name__ == "__main__":
    main()
