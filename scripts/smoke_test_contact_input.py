"""Real pretrained Action Head + Sparsh smoke, without VLM allocation or training run.

Uses synthetic observation/action tensors. Checks real checkpoint parameter parity,
backward, optimizer updates, and adapter save/reload plus a denoising inference call.
It does not validate the full Florence VLM/processor/data/distributed pipeline.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers.feature_extraction_utils import BatchFeature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sparsh", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from gr00t.model.gr00t_n1d7.sensor_loading import unexpected_missing_keys

    torch.set_num_threads(8)
    torch.manual_seed(42)
    config = Gr00tN1d7Config.from_pretrained(args.checkpoint)
    config.contact_input_mode = "moe_soft"
    config.contact_input_diagnostics = True
    config.freeze_pretrained_dit = True
    config.tune_projector = False
    config.tune_vlln = False
    config.tune_tactile_encoder = False
    config.tactile_pretrained_model = args.sparsh
    config.use_tactile_conditioning = True
    config.state_dropout_prob = 0
    config.action_horizon = 2
    config.num_inference_timesteps = 1
    config.action_head_dropout_override = 0
    torch.set_default_dtype(torch.bfloat16)
    head = Gr00tN1d7ActionHead(config).to(args.device).eval()
    torch.set_default_dtype(torch.float32)
    weights = {}
    for file in sorted(args.checkpoint.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith("action_head."):
                    weights[key.removeprefix("action_head.")] = checkpoint.get_tensor(key)
    missing, unexpected = head.load_state_dict(weights, strict=False)
    assert not unexpected
    assert not unexpected_missing_keys(missing, ("contact_input_adapter.", "tactile_encoder."))
    assert all(
        torch.equal(head.state_dict()[key].cpu(), value.to(torch.bfloat16))
        for key, value in weights.items()
    )
    del weights
    from gr00t.model.extension.contact_input_moe import encoder_digest

    original_digest = encoder_digest(head.model)
    device, dtype = args.device, torch.bfloat16
    backbone = BatchFeature(
        {
            "backbone_features": torch.randn(
                1, 4, config.backbone_embedding_dim, device=device, dtype=dtype
            ),
            "backbone_attention_mask": torch.ones(1, 4, device=device, dtype=torch.bool),
            "image_mask": torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.bool),
        }
    )
    inputs = BatchFeature(
        {
            "state": torch.randn(1, 1, config.max_state_dim, device=device, dtype=dtype),
            "action": torch.randn(1, 2, config.max_action_dim, device=device, dtype=dtype),
            "action_mask": torch.ones(1, 2, config.max_action_dim, device=device, dtype=dtype),
            "embodiment_id": torch.zeros(1, device=device, dtype=torch.long),
            "tactile": torch.rand(1, 2, 3, 224, 224, device=device),
        }
    )
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    before = head.contact_input_adapter.injection_scale.detach().clone()
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        output = head(deepcopy(backbone), deepcopy(inputs))
    output["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    gate_grad = float(head.contact_input_adapter.injection_scale.grad.float().norm())
    optimizer.step()
    assert not torch.equal(before, head.contact_input_adapter.injection_scale)
    assert encoder_digest(head.model) == original_digest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(args.output_dir)
    torch.save(head.contact_input_adapter.state_dict(), args.output_dir / "adapter.pt")
    head.contact_input_adapter.load_state_dict(
        torch.load(args.output_dir / "adapter.pt", weights_only=True), strict=True
    )
    del inputs["action"]
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        prediction = head.get_action(deepcopy(backbone), inputs)["action_pred"]
    assert torch.isfinite(prediction).all()
    report = {
        "loss": float(output["loss"].detach()),
        "injection_scale_grad_norm": gate_grad,
        "injection_scale": float(head.contact_input_adapter.injection_scale.detach()),
        "original_dit_sha256": original_digest,
        "inference_shape": list(prediction.shape),
        "adapter_parameters": sum(p.numel() for p in head.contact_input_adapter.parameters()),
        "full_pipeline": "not tested",
        "device": args.device,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
