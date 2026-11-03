"""Four CPU/Gloo ranks: serialized disk reads and error propagation, not GPU validation."""

import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, directory, fail_rank):
    from gr00t.utils import dist_utils
    from gr00t.utils.checkpoint_io import load_torch_checkpoint

    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        GROOT_DEBUG_MODEL_LOAD_IO="1",
        GROOT_MODEL_LOAD_TRACE_DIR=f"{directory}/trace",
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=4
    )
    try:
        # Exercise NCCL->Gloo side-group selection using real Gloo collectives.
        # This does not simulate NCCL kernels or claim a GPU timeout reproduction.
        original_backend = dist.get_backend
        dist.get_backend = lambda group=None: "nccl" if group is None else original_backend(group)
        rng = torch.random.get_rng_state()
        group = dist_utils._serialized_loading_group()
        assert dist_utils._serialized_loading_group() is group
        assert torch.equal(rng, torch.random.get_rng_state())

        def load():
            start = time.monotonic()
            if rank == fail_rank:
                raise ValueError("deliberate loader failure")
            if rank == 2:
                time.sleep(0.1)  # bounded slow-disk stand-in
            state = load_torch_checkpoint(Path(directory) / "weights.pth", weights_only=True)
            end = time.monotonic()
            (Path(directory) / f"body-{rank}.json").write_text(
                json.dumps({"start": start, "end": end})
            )
            return state

        try:
            state = dist_utils.run_serialized_across_ranks(load, label="four-rank checkpoint smoke")
            assert torch.equal(state["weight"], torch.arange(32).float())
            result = "PASS"
        except (RuntimeError, ValueError) as exc:
            if fail_rank < 0:
                raise
            result = f"{type(exc).__name__}: {exc}"
        (Path(directory) / f"result-{rank}.txt").write_text(result)
    finally:
        dist.get_backend = original_backend
        dist.destroy_process_group()


@pytest.mark.serial
@pytest.mark.timeout(120)
@pytest.mark.parametrize("fail_rank", [-1, 2])
def test_four_rank_serialized_checkpoint_loading(
    tmp_path, monkeypatch, serialize_subprocess_spawns, fail_rank
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    torch.save({"weight": torch.arange(32).float()}, tmp_path / "weights.pth")
    mp.spawn(_worker, args=(str(tmp_path), fail_rank), nprocs=4, join=True)
    results = [(tmp_path / f"result-{rank}.txt").read_text() for rank in range(4)]
    if fail_rank < 0:
        assert results == ["PASS"] * 4
        intervals = [json.loads((tmp_path / f"body-{rank}.json").read_text()) for rank in range(4)]
        assert all(a["end"] <= b["start"] for a, b in zip(intervals, intervals[1:]))
    else:
        assert all("deliberate loader failure" in result for result in results)
        assert results[2].startswith("ValueError")
        assert not (tmp_path / "body-3.json").exists()
    for rank in range(4):
        files = list((tmp_path / "trace").glob(f"rank-{rank}-*.jsonl"))
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert any(
            r["phase"] == "serialized.all_reduce"
            and r["event"] == "exit"
            and r["backend"] == "gloo"
            for r in records
        )
        assert any(r["phase"] == "serialized.barrier" and r["event"] == "exit" for r in records)
