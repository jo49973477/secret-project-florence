import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "export_zero3", Path(__file__).resolve().parents[2] / "scripts/export_zero3_checkpoint.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_resolved_config_overrides_base_and_run_fallback(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    provenance = tmp_path / "experiment_cfg"
    provenance.mkdir()
    resolved = {
        "dit_type": "sidecar_conditioned_dit",
        "dit_training_mode": "lora",
        "contact_input_mode": "moe_soft",
        "diffusion_model_cfg": {"num_layers": 16},
        "select_layer": 16,
        "point_encoder_model_config": {"depth": 12},
    }
    (provenance / "final_model_config.json").write_text(json.dumps(resolved))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"select_layer": 12, "original": True}))
    config = exporter.export_config(checkpoint, base)
    assert config["select_layer"] == 16
    assert config["point_encoder_model_config"] == {"depth": 12}
    assert config["original"] is True
    with pytest.raises(FileNotFoundError):
        exporter.artifact(checkpoint, "statistics.json")


@pytest.mark.parametrize(
    "key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
)
def test_restore_rejects_all_loading_errors(key):
    with pytest.raises(RuntimeError, match="Incomplete inference"):
        exporter.validate_loading({key: ["action_head.model.sidecar_stages.0.weight"]})
    exporter.validate_loading({key: []})


def native_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "latest").write_text("global_step10")
    shards = tmp_path / "global_step10"
    shards.mkdir()
    for rank in range(2):
        torch.save(
            {
                "optimizer_state_dict": {
                    "zero_stage": 3,
                    "partition_count": 2,
                    "fp32_flat_groups": [torch.ones(2)],
                }
            },
            shards / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        )
        torch.save(
            {
                "param_shapes": [{"action_head.model.x.lora_A": torch.Size([2, 2])}],
                "buffer_names": [],
                "module": {},
                "shared_params": {},
                "ds_version": "0.17.6",
                "frozen_param_shapes": {"backbone.weight": torch.Size([4])},
                "frozen_param_fragments": {"backbone.weight": torch.ones(2)},
            },
            shards / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
        )
    return shards


def test_inspection_checks_native_shards_without_consolidation(tmp_path):
    native_checkpoint(tmp_path)
    report = exporter.inspect_checkpoint(tmp_path)
    assert report["partitions"] == 2
    assert report["parameters"] == 2
    assert report["lora_parameters"] == 1


def test_missing_rank_rejected(tmp_path):
    shards = native_checkpoint(tmp_path)
    (shards / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
    with pytest.raises(ValueError, match="partition count"):
        exporter.inspect_checkpoint(tmp_path)


def test_missing_frozen_weights_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    shards = native_checkpoint(tmp_path)
    path = shards / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    state = torch.load(path, weights_only=False)
    state.pop("frozen_param_fragments")
    torch.save(state, path)
    with pytest.raises(ValueError, match="Frozen parameter"):
        exporter.inspect_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "name", ["sidecar_stages.weight", "x.lora_B", "input_contact_adapter.router.weight"]
)
def test_native_weight_comparison_detects_random_initialization(name):
    torch = pytest.importorskip("torch")
    native = {name: torch.tensor([0.25, 0.5], dtype=torch.float32)}
    restored = {name: native[name].to(torch.bfloat16)}
    exporter.compare_tensors(restored, native, [name])
    restored[name][0] = 0
    with pytest.raises(RuntimeError, match="differs from native"):
        exporter.compare_tensors(restored, native, [name])
    with pytest.raises(RuntimeError, match="lacks restored"):
        exporter.compare_tensors(restored, {}, [name])


def test_export_passes_directory_and_preserves_inference_artifacts(tmp_path, monkeypatch):
    import sys
    import types

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    output = tmp_path / "inference"
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json", "metadata.json"):
        (checkpoint / name).write_text(json.dumps({"sentinel": name}))
    (checkpoint / "experiment_cfg").mkdir()
    (checkpoint / "experiment_cfg" / "config.yaml").write_text("model: sentinel")
    (checkpoint / "processor").mkdir()
    (checkpoint / "processor" / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(exporter, "export_config", lambda *args: {"dit_training_mode": "lora"})
    monkeypatch.setattr(exporter, "inspect_checkpoint", lambda *args: {"partitions": 2})
    calls = []

    def convert(source, destination, **kwargs):
        calls.append((source, destination, kwargs))
        Path(destination).mkdir()

    fake = types.ModuleType("deepspeed.utils.zero_to_fp32")
    fake.convert_zero_checkpoint_to_fp32_state_dict = convert
    monkeypatch.setitem(sys.modules, "deepspeed.utils.zero_to_fp32", fake)
    monkeypatch.setattr(exporter, "verify_export", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(sys, "argv", ["export", str(checkpoint), str(output)])
    exporter.main()
    assert calls == [
        (
            str(checkpoint),
            str(output),
            {"safe_serialization": True, "exclude_frozen_parameters": False},
        )
    ]
    for name in ("statistics.json", "processor_config.json", "embodiment_id.json", "metadata.json"):
        assert (checkpoint / name).read_bytes() == (output / name).read_bytes()
    assert (output / "processor/tokenizer.json").is_file()
    assert (output / "experiment_cfg/config.yaml").is_file()
    with pytest.raises(SystemExit):
        exporter.main()  # Existing exports must never be overwritten.
    assert len(calls) == 1


def test_installed_deepspeed_restores_trainable_and_frozen_partitions(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    native_checkpoint(tmp_path)
    restored = get_fp32_state_dict_from_zero_checkpoint(str(tmp_path))
    assert torch.equal(restored["backbone.weight"], torch.ones(4))
    assert torch.equal(restored["action_head.model.x.lora_A"], torch.ones(2, 2))
