from gr00t.model.gr00t_n1d7.dit_integrity import compare_shared_tensors, overall_status
import torch


def test_shared_weights_pass_and_new_adapter_is_excluded():
    ref = {
        "action_head.model.block.weight": torch.ones(2),
        "action_head.model.point_gates": torch.zeros(1),
    }
    candidate = {
        "action_head.model.block.weight": torch.ones(2),
        "action_head.model.point_gates": torch.ones(1),
    }
    result = compare_shared_tensors(ref, candidate)
    assert result["status"] == "PASS"
    assert result["expected_tensors"] == result["matched_tensors"] == 1


def test_changed_or_missing_original_weight_fails():
    ref = {"action_head.model.block.weight": torch.ones(2)}
    changed = compare_shared_tensors(ref, {"action_head.model.block.weight": torch.zeros(2)})
    missing = compare_shared_tensors(ref, {})
    assert changed["status"] == "FAIL"
    assert changed["mismatched"][0]["reason"] == "value"
    assert missing["status"] == "FAIL"
    assert missing["missing"] == ["action_head.model.block.weight"]


def test_overall_pass_fail_and_skip_are_distinguished():
    assert overall_status(["PASS", "PASS"]) == "PASS"
    assert overall_status(["PASS", "SKIP"]) == "INCOMPLETE"
    assert overall_status(["SKIP", "FAIL"]) == "FAIL"


def test_reference_missing_candidate_original_fails():
    result = compare_shared_tensors({}, {"action_head.model.block.weight": torch.ones(2)})
    assert result["status"] == "FAIL"
    assert result["expected_tensors"] == 1


def test_intentional_bfloat16_cast_is_exact():
    ref = {"action_head.model.block.weight": torch.tensor([1.123456, -3.142])}
    assert (
        compare_shared_tensors(ref, {k: v.bfloat16() for k, v in ref.items()})["status"] == "PASS"
    )


def test_shared_loader_validation():
    from gr00t.model.gr00t_n1d7.dit_integrity import validate_loading_info
    from gr00t.model.gr00t_n1d7.sensor_loading import MULTIMODAL_ADAPTER_PREFIXES

    assert (
        validate_loading_info(
            {"missing_keys": ["action_head.model.point_gates"]}, MULTIMODAL_ADAPTER_PREFIXES
        )["status"]
        == "PASS"
    )
    assert (
        validate_loading_info(
            {"missing_keys": ["action_head.model.transformer_blocks.0.weight"]},
            MULTIMODAL_ADAPTER_PREFIXES,
        )["status"]
        == "FAIL"
    )


def test_real_dit_forward_helper_small_fixture():
    from gr00t.model.extension.multimodal_dit import MultiModalConditionedDiT
    from gr00t.model.gr00t_n1d7.dit_integrity import forward_parity
    from gr00t.model.modules.dit import AlternateVLDiT

    config = dict(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=6,
    )
    original, candidate = AlternateVLDiT(**config), MultiModalConditionedDiT(**config)
    candidate.load_state_dict(original.state_dict(), strict=False)
    report = forward_parity(original, candidate, config)
    assert report["status"] == "PASS"
    assert len(report["cases"]) == 3


def test_script_with_local_small_checkpoint(tmp_path, monkeypatch):
    """Exercise the real CLI path using a labelled small fixture, not GR00T validation."""
    import importlib.util
    import json
    from pathlib import Path
    import sys

    from gr00t.model.modules.dit import AlternateVLDiT
    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location(
        "verify_dit", Path(__file__).resolve().parents[3] / "scripts/verify_pretrained_dit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=True,
    )
    dit = AlternateVLDiT(**config, cross_attention_dim=6)
    checkpoint = tmp_path / "fixture"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"diffusion_model_cfg": config, "backbone_embedding_dim": 6})
    )
    state = {"action_head.model." + k: v for k, v in dit.state_dict().items()}
    save_file(state, str(checkpoint / "model.safetensors"))
    report_path = tmp_path / "report.json"
    argv = [
        "verify_pretrained_dit.py",
        "--reference-checkpoint",
        str(checkpoint),
        "--report",
        str(report_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 2  # Forward explicitly skipped
    assert json.loads(report_path.read_text())["forward_check"]["status"] == "SKIP"
    monkeypatch.setattr(sys, "argv", argv + ["--forward-parity"])
    assert module.main() == 0
    assert json.loads(report_path.read_text())["overall"] == "PASS"
    state.pop(next(iter(state)))
    save_file(state, str(checkpoint / "model.safetensors"))
    assert module.main() == 1
    assert json.loads(report_path.read_text())["weight_check"]["status"] == "FAIL"


def test_optional_real_checkpoint(tmp_path, monkeypatch):
    import importlib.util
    import os
    from pathlib import Path
    import sys

    import pytest

    checkpoint = os.environ.get("GR00T_DIT_INTEGRATION_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set GR00T_DIT_INTEGRATION_CHECKPOINT to a local real pretrained snapshot")
    spec = importlib.util.spec_from_file_location(
        "verify_real_dit", Path(__file__).resolve().parents[3] / "scripts/verify_pretrained_dit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_pretrained_dit.py",
            "--reference-checkpoint",
            checkpoint,
            "--forward-parity",
            "--report",
            str(tmp_path / "real.json"),
        ],
    )
    assert module.main() == 0
