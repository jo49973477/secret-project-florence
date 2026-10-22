"""Small deterministic CPU checks; no sensor downloads or GR00T checkpoint."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from gr00t.model.extension.contact_input_moe import (
    ContactInputAdapter,
    ContactSurpriseExpert,
    DirectContactExpert,
    NoOpContactExpert,
    encoder_digest,
)
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.sensor_loading import checkpoint_sensor_layout, unexpected_missing_keys
import pytest
from tests.gr00t.model.test_action_head import (
    _make_action_input,
    _make_backbone_output,
    _small_config,
)
import torch
from torch import nn


def sample(dtype=torch.float32):
    torch.manual_seed(4)
    return (
        torch.randn(2, 4, 16, dtype=dtype),
        torch.randn(2, 7, 16, dtype=dtype),
        torch.randn(2, 1, 3, dtype=dtype),
    )


@pytest.mark.parametrize("mode", ["direct", "scalar_gate", "moe_soft"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parity_gradients_optimizer(mode, dtype):
    a, t, s = sample(dtype)
    adapter = ContactInputAdapter(16, 3, mode, 8, 2, diagnostics=True).to(dtype)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.01, weight_decay=0)
    before = deepcopy(adapter.state_dict())
    contact = adapter.prepare_contact(t, s)
    out = adapter(a, contact, s)
    assert torch.equal(a, out)
    out.float().square().mean().backward()
    assert torch.isfinite(adapter.injection_scale.grad).all()
    assert adapter.injection_scale.grad.abs().sum() > 0
    assert all(
        p.grad is None or p.grad.count_nonzero() == 0
        for n, p in adapter.named_parameters()
        if n != "injection_scale"
    )
    opt.step()
    assert not torch.equal(adapter.injection_scale, before["injection_scale"])
    assert torch.equal(adapter.direct_expert.film.weight, before["direct_expert.film.weight"])
    opt.zero_grad()
    out = adapter(a, adapter.prepare_contact(t, s), s)
    out.float().square().mean().backward()
    for module in (adapter.direct_expert, adapter.router, adapter.scalar_gate):
        if module is not None:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    opt.step()
    assert not torch.equal(adapter.direct_expert.film.weight, before["direct_expert.film.weight"])
    assert adapter.last_diagnostics["contact_input_residual_ratio"] > 0


def test_expert_shapes_noop_and_surprise():
    a, t, _ = sample()
    assert torch.equal(NoOpContactExpert()(a), torch.zeros_like(a))
    for expert in (DirectContactExpert(16, 16), ContactSurpriseExpert(16, 16)):
        assert expert(a, t).shape == a.shape


def test_routing_mask_and_no_leakage():
    a, t, s = sample()
    adapter = ContactInputAdapter(16, 3, contact_dim=8, num_tokens=2)
    contact = adapter.prepare_contact(t, s)
    p = adapter.routing_probabilities(s, *contact[:2])
    assert torch.equal(p[:, 2], torch.zeros(2))
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    adapter(a, contact, s)
    adapter(a * 100 + 1, contact, s)
    assert torch.equal(p, adapter.routing_probabilities(s, *contact[:2]))
    assert not any("surprise" in key for key in adapter.last_diagnostics)


@pytest.mark.parametrize("empty", [False, True])
def test_masked_contact_and_action(empty):
    a, t, s = sample()
    adapter = ContactInputAdapter(16, 3, contact_dim=8, num_tokens=2)
    adapter.injection_scale.data.fill_(1)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    if empty:
        t, mask = t[:, :0], mask[:, :0]
    else:
        t.fill_(float("nan"))
    contact = adapter.prepare_contact(t, s, mask)
    assert torch.isfinite(contact[0]).all()
    assert torch.equal(adapter(a, contact, s), a)
    p = adapter.routing_probabilities(s, *contact[:2])
    assert torch.equal(p, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    a, t, s = sample()
    contact = adapter.prepare_contact(t, s)
    action_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    out = adapter(a, contact, s, action_mask)
    assert torch.equal(out[~action_mask], a[~action_mask])


def test_predictor_requires_aligned_bundle(tmp_path):
    a, t, s = sample()
    adapter = ContactInputAdapter(16, 3, contact_dim=8, num_tokens=2, use_surprise=True)
    with pytest.raises(RuntimeError, match="validated"):
        adapter.prepare_contact(t, s)
    encoder = nn.Linear(3, 16)
    path = tmp_path / "predictor.pt"
    bundle = {
        "metadata": {
            "format": "florence-contact-state-predictor-v1",
            "num_tokens": 2,
            "contact_dim": 8,
            "encoder_sha256": encoder_digest(encoder),
            "training_provenance": "synthetic unit fixture, not a real predictor",
            "validation_report": "synthetic shape fixture",
        },
        "predictor": adapter.predictor.state_dict(),
        "projector": adapter.projector.state_dict(),
    }
    torch.save(bundle, path)
    adapter.load_predictor(path, encoder)
    adapter.injection_scale.data.fill_(0.1)
    contact = adapter.prepare_contact(t, s)
    p = adapter.routing_probabilities(s, contact[0], contact[1], contact[2])
    assert (p > 0).all()
    adapter(a, contact, s).square().mean().backward()
    assert adapter.surprise_expert.film.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.predictor.parameters())
    bundle["metadata"]["encoder_sha256"] = "wrong"
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="mismatch"):
        adapter.load_predictor(path, encoder)


class TinySensor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(3, width)

    def forward(self, images):
        return self.proj(images.mean((-1, -2)))


@pytest.mark.parametrize("mode", ["direct", "scalar_gate", "moe_soft"])
def test_head_train_inference_parity_reload_freezing(mode, tmp_path):
    config = _small_config(
        dit_type="alternate_vl_dit",
        contact_input_mode=mode,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        freeze_pretrained_dit=True,
    )
    with patch(
        "gr00t.model.gr00t_n1d7.gr00t_n1d7.build_tactile_encoder",
        side_effect=lambda *args, **kw: TinySensor(64),
    ):
        head = Gr00tN1d7ActionHead(config).eval()
        base = Gr00tN1d7ActionHead(_small_config(dit_type="alternate_vl_dit")).eval()
        missing, unexpected = head.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected
        assert not unexpected_missing_keys(missing, ("contact_input_adapter.", "tactile_encoder."))
        assert all(
            torch.equal(v, head.model.state_dict()[k]) for k, v in base.model.state_dict().items()
        )
        assert all(not p.requires_grad for p in head.model.parameters())
        inp = _make_action_input(config)
        inp["tactile"] = torch.randn(2, 2, 3, 4, 4)
        inp["action_mask"][:, -1] = 0
        backbone = _make_backbone_output(config)
        backbone["backbone_attention_mask"] = backbone["backbone_attention_mask"].bool()
        torch.manual_seed(9)
        reference = base(deepcopy(backbone), deepcopy(inp))
        torch.manual_seed(9)
        result = head(deepcopy(backbone), deepcopy(inp))
        assert torch.equal(result["loss"], reference["loss"])
        result["loss"].backward()
        assert head.contact_input_adapter.injection_scale.grad.abs().sum() > 0
        before = deepcopy(head.model.state_dict())
        opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.01)
        opt.step()
        assert all(torch.equal(v, head.model.state_dict()[k]) for k, v in before.items())
        assert head.contact_input_adapter.injection_scale.abs().sum() > 0
        del inp["action"]
        with (
            patch.object(
                head.tactile_encoder, "forward", wraps=head.tactile_encoder.forward
            ) as sensor,
            patch.object(
                head.contact_input_adapter, "forward", wraps=head.contact_input_adapter.forward
            ) as adapter,
        ):
            out = head.get_action(deepcopy(backbone), deepcopy(inp))["action_pred"]
            assert sensor.call_count == 1
            assert adapter.call_count == config.num_inference_timesteps
        assert torch.count_nonzero(out[:, -1]) == 0
        config.save_pretrained(tmp_path)
        torch.save(head.state_dict(), tmp_path / "head.pt")
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

        restored = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(tmp_path)).eval()
        restored.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True), strict=True)
        torch.manual_seed(10)
        first = head.get_action(deepcopy(backbone), deepcopy(inp))["action_pred"]
        torch.manual_seed(10)
        second = restored.get_action(deepcopy(backbone), deepcopy(inp))["action_pred"]
        assert torch.equal(first, second)


def test_checkpoint_layout_strictness():
    c = SimpleNamespace(
        dit_type="alternate_vl_dit", contact_input_mode="moe_soft", use_tactile_conditioning=True
    )
    assert checkpoint_sensor_layout(c) == (False, False, True)
    assert unexpected_missing_keys(
        ["action_head.model.original.weight"], ("action_head.contact_input_adapter.",)
    )


def test_real_optimizer_ownership(tmp_path):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import TrainingArguments

    config = _small_config(
        dit_type="alternate_vl_dit",
        contact_input_mode="moe_soft",
        tactile_encoder_deferred_bootstrap=True,
        freeze_pretrained_dit=True,
    )
    model = nn.Module()
    model.config = config
    model.backbone = nn.Linear(2, 2)
    model.backbone.requires_grad_(False)
    model.action_head = Gr00tN1d7ActionHead(config)
    trainer = object.__new__(Gr00tTrainer)
    trainer.args = TrainingArguments(
        output_dir=str(tmp_path), report_to="none", optim="adamw_torch"
    )
    trainer.vlm_learning_rate = 1e-4
    trainer.action_head_learning_rate = 1e-4
    trainer.point_encoder_learning_rate = None
    trainer.tactile_encoder_learning_rate = None
    groups = trainer._create_split_optimizer_groups(model)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_hf_original_loading_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(3, 3)

    original = tmp_path / "original"
    adapted = tmp_path / "adapted"
    with (
        patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls", return_value=Backbone),
        patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.Gr00tN1d7DataCollator"),
    ):
        cfg = _small_config(dit_type="alternate_vl_dit")
        base = Gr00tN1d7(cfg)
        base.save_pretrained(original)
        head_weights = deepcopy(base.action_head.model.state_dict())
        model, info = Gr00tN1d7.from_pretrained(
            original,
            contact_input_mode="moe_soft",
            contact_hidden_dim=8,
            tactile_encoder_deferred_bootstrap=True,
            freeze_pretrained_dit=True,
            output_loading_info=True,
        )
        assert not info["unexpected_keys"] and not info["mismatched_keys"]
        assert not unexpected_missing_keys(
            info["missing_keys"], ("action_head.contact_input_adapter.",)
        )
        assert model.action_head.contact_input_adapter.injection_scale.item() == 0
        assert torch.isfinite(model.action_head.contact_input_adapter.projector.queries).all()
        assert model.action_head.contact_input_adapter.projector.queries.std() > 0
        assert all(
            torch.equal(v, model.action_head.model.state_dict()[k]) for k, v in head_weights.items()
        )
        with torch.no_grad():
            model.action_head.contact_input_adapter.injection_scale.fill_(0.25)
        model.save_pretrained(adapted)
        reloaded, info = Gr00tN1d7.from_pretrained(adapted, output_loading_info=True)
        assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
        assert reloaded.action_head.contact_input_adapter.injection_scale.item() == 0.25
        assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items())


def test_shared_sensor_initialization_across_modes():
    weights = []
    for mode in ("direct", "scalar_gate", "moe_soft"):
        torch.manual_seed(42)
        with patch(
            "gr00t.model.gr00t_n1d7.gr00t_n1d7.build_tactile_encoder",
            side_effect=lambda *a, **kw: TinySensor(64),
        ):
            head = Gr00tN1d7ActionHead(
                _small_config(dit_type="alternate_vl_dit", contact_input_mode=mode)
            )
        weights.append(head.tactile_encoder.proj.weight.detach().clone())
    assert all(torch.equal(weights[0], w) for w in weights[1:])


def test_florence_loader_strict_validation(tmp_path, monkeypatch):
    from gr00t.configs.base_config import Config
    import gr00t.model.gr00t_n1d7.setup as setup
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    config = Config()
    config.model = _small_config(dit_type="alternate_vl_dit", contact_input_mode="moe_soft")
    config.training.start_from_checkpoint = "synthetic-original"
    pipeline = object.__new__(Gr00tN1d7Pipeline)
    pipeline.config = config
    pipeline.save_cfg_dir = tmp_path
    pipeline.transformers_loading_kwargs = {}
    model = nn.Module()
    model.config = config.model
    model.action_head = Gr00tN1d7ActionHead(
        _small_config(
            dit_type="alternate_vl_dit",
            contact_input_mode="moe_soft",
            tactile_encoder_deferred_bootstrap=True,
        )
    )
    model.action_head.bootstrap_deferred_sensor_backbones = lambda: None
    monkeypatch.setattr(
        setup.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: _small_config(dit_type="alternate_vl_dit"),
    )
    info = {
        "missing_keys": ["action_head.contact_input_adapter.injection_scale"],
        "unexpected_keys": [],
        "mismatched_keys": [],
    }
    monkeypatch.setattr(
        setup.AutoModel, "from_pretrained", lambda *a, **kw: (model, deepcopy(info))
    )
    assert pipeline._create_model() is model
    for field, key in (
        ("missing_keys", "action_head.model.proj_out_1.weight"),
        ("unexpected_keys", "action_head.some_unexpected.weight"),
        ("mismatched_keys", "action_head.contact_input_adapter.projector.projection.weight"),
    ):
        info[field].append(key)
        with pytest.raises(RuntimeError, match="Checkpoint weight mismatch"):
            pipeline._create_model()
        info[field].pop()


def test_zero3_local_gradient_diagnostics(monkeypatch):
    import sys
    from types import ModuleType

    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import Trainer

    adapter = ContactInputAdapter(16, 3, contact_dim=8, diagnostics=True)
    model = nn.Module()
    model.contact_input_adapter = adapter
    for parameter in adapter.parameters():
        parameter.ds_id = 1
    fake = ModuleType("deepspeed.utils")
    fake.safe_get_local_grad = lambda parameter: torch.ones(2)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", fake)
    monkeypatch.setattr(Trainer, "training_step", lambda *args, **kwargs: torch.tensor(1.0))
    trainer = object.__new__(Gr00tTrainer)
    trainer.training_step(model, {})
    assert trainer._gradient_diagnostics["contact_input_injection_scale_grad"] == 2
    assert trainer._gradient_diagnostics["contact_router_grad_norm"] > 0
    assert "contact_surprise_expert_grad_norm" not in trainer._gradient_diagnostics


def test_lora_validation_accepts_explicit_frozen_dit():
    from gr00t.model.gr00t_n1d7.setup import log_parameter_summary

    config = _small_config(
        dit_type="alternate_vl_dit",
        contact_input_mode="moe_soft",
        freeze_pretrained_dit=True,
        tune_projector=False,
        tune_vlln=False,
        tactile_encoder_deferred_bootstrap=True,
        use_lora=True,
    )
    model = nn.Module()
    model.config = config
    model.backbone = nn.Module()
    model.backbone.lora_A = nn.Parameter(torch.ones(2))
    model.action_head = Gr00tN1d7ActionHead(config)
    log_parameter_summary(model)


def test_embedded_surprise_checkpoint_reload(tmp_path):
    """Synthetic fixture only: tests checkpoint mechanics, not predictor quality."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    cfg = _small_config(
        dit_type="alternate_vl_dit",
        contact_input_mode="moe_soft",
        contact_moe_use_surprise=True,
        contact_hidden_dim=8,
        contact_num_tokens=2,
        freeze_pretrained_dit=True,
    )
    with patch(
        "gr00t.model.gr00t_n1d7.gr00t_n1d7.build_tactile_encoder",
        side_effect=lambda *a, **kw: TinySensor(64),
    ):
        head = Gr00tN1d7ActionHead(cfg).eval()
        adapter = head.contact_input_adapter
        bundle = {
            "metadata": {
                "format": "florence-contact-state-predictor-v1",
                "num_tokens": 2,
                "contact_dim": 8,
                "encoder_sha256": encoder_digest(head.tactile_encoder),
                "training_provenance": "synthetic unit fixture",
                "validation_report": "checkpoint mechanics only",
            },
            "predictor": adapter.predictor.state_dict(),
            "projector": adapter.projector.state_dict(),
        }
        path = tmp_path / "fixture.pt"
        torch.save(bundle, path)
        adapter.load_predictor(path, head.tactile_encoder)
        cfg.contact_predictor_embedded = True
        adapter.injection_scale.data.fill_(0.1)
        cfg.save_pretrained(tmp_path)
        torch.save(head.state_dict(), tmp_path / "head.pt")
        restored = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(tmp_path)).eval()
        restored.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True), strict=True)
        assert restored.contact_input_adapter.predictor_ready
        assert not any(p.requires_grad for p in restored.tactile_encoder.parameters())
        assert not any(
            p.requires_grad for p in restored.contact_input_adapter.predictor.parameters()
        )
        inp = _make_action_input(cfg)
        inp["tactile"] = torch.randn(2, 2, 3, 4, 4)
        del inp["action"]
        backbone = _make_backbone_output(cfg)
        backbone["backbone_attention_mask"] = backbone["backbone_attention_mask"].bool()
        torch.manual_seed(1)
        before = head.get_action(deepcopy(backbone), deepcopy(inp))["action_pred"]
        torch.manual_seed(1)
        after = restored.get_action(deepcopy(backbone), deepcopy(inp))["action_pred"]
        assert torch.equal(before, after)
