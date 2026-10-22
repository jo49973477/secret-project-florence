"""Exercise offline validation/ablation control flow without VLM allocation."""

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def test_evaluation_conditions_and_complete_windows(tmp_path, monkeypatch):
    import gr00t.data.dataset.lerobot_episode_loader as loader_module
    import gr00t.data.dataset.sharded_single_step_dataset as step_module
    import gr00t.policy.gr00t_policy as policy_module
    import scripts.evaluate_contact_input as evaluation

    configs = {
        "tactile": SimpleNamespace(delta_indices=[-1, 0]),
        "action": SimpleNamespace(delta_indices=[0, 1, 2, 3]),
    }

    class Adapter(nn.Module):
        def forward(self, actions, contact):
            return actions + contact

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = nn.Module()
            self.action_head.contact_input_adapter = Adapter()

        def forward(self, point):
            contact = torch.tensor(np.asarray(point.tactile["rgb"]).mean() / 255)
            pred = self.action_head.contact_input_adapter(torch.ones(1, 2, 2), contact)
            return {
                "loss": pred.square().mean() + torch.rand(()),
                "action_mask": torch.ones(1, 2, 2),
            }

    policy = SimpleNamespace(
        model=Model(),
        modality_configs=configs,
        processor=lambda messages: messages[0]["content"],
        collate_fn=lambda points: points[0],
    )
    monkeypatch.setattr(policy_module, "Gr00tPolicy", lambda **kw: policy)

    class Loader:
        def __init__(self, *args):
            pass

        def __getitem__(self, episode):
            return list(range(10))

    monkeypatch.setattr(loader_module, "LeRobotEpisodeLoader", Loader)
    original_points = []

    def extract(trajectory, step, configs, embodiment):
        assert 1 <= step < 7  # no future/negative observations beyond episode boundaries
        point = SimpleNamespace(tactile={"rgb": [np.full((2, 2, 3), step, dtype=np.uint8)] * 2})
        original_points.append(point)
        return point

    monkeypatch.setattr(step_module, "extract_step_data", extract)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_contact_input.py",
            "--checkpoint",
            "synthetic",
            "--dataset-path",
            str(tmp_path),
            "--episode-ids",
            "0",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--conditions",
            "enabled",
            "off",
            "shuffle",
            "noise",
            "enabled",
            "--output",
            str(output),
        ],
    )
    evaluation.main()
    report = json.loads(output.read_text())
    assert set(report["masked_flow_matching_loss"]) == {"enabled", "off", "shuffle", "noise"}
    assert len(report["samples"]) == 10
    enabled = [row["loss"] for row in report["samples"] if row["condition"] == "enabled"]
    assert enabled[:2] == enabled[2:]  # same flow noise for repeated conditions
    assert np.asarray(original_points[0].tactile["rgb"]).max() == 1
