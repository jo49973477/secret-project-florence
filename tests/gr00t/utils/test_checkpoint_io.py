"""Exact local checkpoint values and observation-only load diagnostics."""

import json

from gr00t.utils.checkpoint_io import load_torch_checkpoint
from gr00t.utils.model_load_diagnostics import model_load_phase, trace_checkpoint_loading
import pytest
import torch


@pytest.mark.parametrize("zip_format", [True, False])
def test_mmap_and_legacy_format_are_identical(tmp_path, zip_format):
    path = tmp_path / "weights.pth"
    source = {
        "config": {"width": 4},
        "state_dict": {"weight": torch.arange(32).reshape(8, 4).float()},
    }
    torch.save(source, path, _use_new_zipfile_serialization=zip_format)
    legacy = load_torch_checkpoint(path, weights_only=True, memory_map=False)
    mapped = load_torch_checkpoint(path, weights_only=True)
    assert mapped["config"] == legacy["config"]
    assert torch.equal(mapped["state_dict"]["weight"], legacy["state_dict"]["weight"])
    # File-backed tensors remain live after the loader function exits.
    assert mapped["state_dict"]["weight"].sum().item() == 496


def test_mmap_option_only_for_zip_and_never_retries_failed_load(tmp_path, monkeypatch):
    path = tmp_path / "weights.pth"
    torch.save({"weight": torch.ones(4)}, path)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("invalid checkpoint")

    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(ValueError, match="invalid checkpoint"):
        load_torch_checkpoint(path, weights_only=True)
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]


def test_trace_records_actual_shards_and_restores_loaders(tmp_path, monkeypatch):
    from transformers import modeling_utils

    monkeypatch.setenv("GROOT_DEBUG_MODEL_LOAD_IO", "1")
    monkeypatch.setenv("GROOT_MODEL_LOAD_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.setenv("RANK", "2")
    path = tmp_path / "weights.pth"
    torch.save({"weight": torch.ones(4)}, path)
    originals = torch.load, modeling_utils.load_state_dict, modeling_utils.load_shard_file
    with trace_checkpoint_loading():
        with model_load_phase("test.materialize", checkpoint=path):
            state = modeling_utils.load_state_dict(str(path), weights_only=True)
    assert torch.equal(state["weight"], torch.ones(4))
    assert (torch.load, modeling_utils.load_state_dict, modeling_utils.load_shard_file) == originals
    records = [
        json.loads(line)
        for file in (tmp_path / "trace").glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]
    exits = [record for record in records if record["event"] == "exit"]
    assert {r["phase"] for r in exits} >= {
        "torch.load",
        "transformers.load_state_dict",
        "test.materialize",
    }
    for record in exits:
        assert record["rank"] == "2"
        assert record["file_size_bytes"] == path.stat().st_size
        assert record["elapsed_seconds"] >= 0
        assert record["read_bytes_delta"] >= 0 and record["rchar_delta"] >= 0
        assert record["peak_kib"] >= record["rss_kib"] > 0
        assert record["utc"] and record["pid"]


def test_trace_exception_is_recorded_and_original_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("GROOT_DEBUG_MODEL_LOAD_IO", "1")
    monkeypatch.setenv("GROOT_MODEL_LOAD_TRACE_DIR", str(tmp_path))
    original = torch.load
    with pytest.raises(FileNotFoundError):
        with trace_checkpoint_loading():
            torch.load(tmp_path / "missing.pth", weights_only=True)
    assert torch.load is original
    records = [
        json.loads(line)
        for file in tmp_path.glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]
    assert records[-1]["error"] == "FileNotFoundError"


def test_disabled_trace_does_not_patch_or_collect_counters(monkeypatch):
    monkeypatch.delenv("GROOT_DEBUG_MODEL_LOAD_IO", raising=False)

    def forbidden():
        raise AssertionError("diagnostics disabled")

    monkeypatch.setattr("gr00t.utils.model_load_diagnostics.model_load_snapshot", forbidden)
    original = torch.load
    with trace_checkpoint_loading(), model_load_phase("no-op"):
        assert torch.load is original


def test_mmap_comparison_switch_does_not_change_unpickle_policy(tmp_path, monkeypatch):
    path = tmp_path / "weights.pth"
    torch.save({"weight": torch.ones(4)}, path)
    monkeypatch.setenv("GROOT_MODEL_LOAD_MMAP", "0")
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(torch, "load", observed)
    load_torch_checkpoint(path, weights_only=False)
    assert calls == [{"map_location": "cpu", "weights_only": False}]
