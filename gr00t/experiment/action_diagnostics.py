"""Detached forward statistics; arithmetic never participates in optimization."""

import torch


@torch.no_grad()
def action_diagnostics(prediction, target, mask, total, action, contact, weight, warmup=False):
    pred, target, mask = prediction.detach().float(), target.detach().float(), mask.float()
    count = mask.sum().clamp_min(1)
    pm, tm = (pred * mask).sum() / count, (target * mask).sum() / count
    zero = (target.square() * mask).sum() / count
    weighted = contact.detach().float() * weight
    contributing_action = torch.zeros_like(action.detach()) if warmup else action.detach()
    values = {
        "loss/total": total.detach(),
        "loss/action": contributing_action,
        "loss/action_observed": action.detach(),
        "loss/contact_raw": contact.detach(),
        "loss/contact_weighted": weighted,
        "loss/zero_prediction_mse": zero,
        "action/predicted_velocity_mean": pm,
        "action/target_velocity_mean": tm,
        "action/predicted_velocity_std": ((pred - pm).square() * mask).sum().div(count).sqrt(),
        "action/target_velocity_std": ((target - tm).square() * mask).sum().div(count).sqrt(),
        "action/valid_action_ratio": mask.mean(),
        "action/valid_action_count": mask.sum(),
        "action/valid_dimension_count": mask.bool().any(dim=-2).float().sum(dim=-1).mean(),
        "action/valid_timestep_ratio": mask.bool().any(dim=-1).float().mean(),
        "loss/warmup": total.new_tensor(float(warmup)),
    }
    # Preserve the historical un-namespaced metrics.
    for key in ("predicted_velocity_std", "target_velocity_std", "valid_action_ratio"):
        values[key] = values[f"action/{key}"]
    values["zero_prediction_mse"] = zero
    return values


def loss_ratios(values):
    """Ratios of the same reduced window means, rather than means of ratios."""
    zero, total = values.get("loss/zero_prediction_mse"), values.get("loss/total")
    if zero is not None and zero > 0:
        values["loss/action_to_zero_ratio"] = values["loss/action"] / zero
    if total is not None and total != 0:
        values["loss/contact_fraction"] = values["loss/contact_weighted"] / total
    return values
