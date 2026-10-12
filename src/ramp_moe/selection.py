"""Budget-controlled selection and Qwen3-MoE joint masking."""

from __future__ import annotations

import torch


def _integer_keep_count(item_count: int, prune_ratio: float) -> int:
    if not 0 < prune_ratio < 1:
        raise ValueError("prune_ratio must be strictly between zero and one")
    exact_count = item_count * (1 - prune_ratio)
    keep_count = round(exact_count)
    if abs(exact_count - keep_count) > 1e-8:
        raise ValueError(
            "prune_ratio does not produce an integer retained-channel budget"
        )
    return keep_count


def select_camera_keep_mask(
    importance: torch.Tensor,
    *,
    prune_ratio: float = 0.5,
    allocation: str = "global",
    min_keep_per_expert: int = 0,
) -> torch.Tensor:
    """Select highest CAMERA energies under global or per-expert budgets."""

    if importance.ndim != 2 or min(importance.shape) == 0:
        raise ValueError("importance must have shape [experts, channels]")
    if not torch.isfinite(importance).all():
        raise ValueError("importance must contain only finite values")
    expert_count, channel_count = importance.shape
    keep_mask = torch.zeros_like(importance, dtype=torch.bool)
    if isinstance(min_keep_per_expert, bool) or min_keep_per_expert < 0:
        raise ValueError("min_keep_per_expert must be a non-negative integer")
    if min_keep_per_expert > channel_count:
        raise ValueError("min_keep_per_expert cannot exceed channel count")
    if allocation != "global" and min_keep_per_expert:
        raise ValueError("min_keep_per_expert is only supported for global allocation")

    if allocation == "global":
        keep_count = _integer_keep_count(importance.numel(), prune_ratio)
        required = expert_count * min_keep_per_expert
        if required > keep_count:
            raise ValueError("minimum expert budget exceeds the global keep budget")
        if min_keep_per_expert:
            expert_order = torch.argsort(
                importance, dim=1, descending=True, stable=True
            )
            keep_mask.scatter_(
                1, expert_order[:, :min_keep_per_expert], True
            )
        remaining = keep_count - int(keep_mask.sum().item())
        if remaining == 0:
            return keep_mask
        eligible = importance.flatten().clone()
        eligible[keep_mask.flatten()] = -torch.inf
        order = torch.argsort(
            eligible, descending=True, stable=True
        )
        keep_mask.flatten()[order[:remaining]] = True
        return keep_mask
    if allocation == "uniform":
        keep_count = _integer_keep_count(channel_count, prune_ratio)
        order = torch.argsort(importance, dim=1, descending=True, stable=True)
        keep_mask.scatter_(1, order[:, :keep_count], True)
        if not torch.all(keep_mask.sum(dim=1) == keep_count):
            raise RuntimeError("failed to enforce the uniform per-expert budget")
        return keep_mask
    raise ValueError("allocation must be either 'global' or 'uniform'")


def select_bounded_camera_keep_mask(
    importance: torch.Tensor,
    *,
    reference_importance: torch.Tensor,
    prune_ratio: float = 0.5,
    max_width_delta: int,
    min_keep_per_expert: int = 0,
) -> torch.Tensor:
    """Select globally while bounding expert widths around a reference allocation.

    The reference signal defines only the expert-level width vector.  Channels
    are re-ranked entirely by ``importance`` inside the resulting trust region.
    """

    if importance.ndim != 2 or min(importance.shape) == 0:
        raise ValueError("importance must have shape [experts, channels]")
    reference_importance = torch.as_tensor(
        reference_importance, dtype=importance.dtype, device=importance.device
    )
    if reference_importance.shape != importance.shape:
        raise ValueError("reference_importance must match importance shape")
    if not torch.isfinite(importance).all() or not torch.isfinite(
        reference_importance
    ).all():
        raise ValueError("importance tensors must contain only finite values")
    if isinstance(max_width_delta, bool) or not isinstance(max_width_delta, int):
        raise ValueError("max_width_delta must be a non-negative integer")
    if max_width_delta < 0:
        raise ValueError("max_width_delta must be a non-negative integer")

    expert_count, channel_count = importance.shape
    reference_mask = select_camera_keep_mask(
        reference_importance,
        prune_ratio=prune_ratio,
        allocation="global",
        min_keep_per_expert=min_keep_per_expert,
    )
    reference_widths = reference_mask.sum(dim=1)
    lower = (reference_widths - max_width_delta).clamp_min(min_keep_per_expert)
    upper = (reference_widths + max_width_delta).clamp_max(channel_count)
    keep_count = _integer_keep_count(importance.numel(), prune_ratio)
    if lower.sum() > keep_count or upper.sum() < keep_count:
        raise ValueError("max_width_delta produces infeasible expert bounds")

    expert_order = torch.argsort(importance, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(expert_order)
    rank_values = torch.arange(channel_count, device=importance.device).expand(
        expert_count, -1
    )
    ranks.scatter_(1, expert_order, rank_values)
    mandatory = ranks < lower.unsqueeze(1)
    eligible = ranks < upper.unsqueeze(1)
    keep_mask = mandatory.clone()
    remaining = keep_count - int(keep_mask.sum().item())
    scores = importance.flatten().clone()
    scores[(~eligible | mandatory).flatten()] = -torch.inf
    selected = torch.argsort(scores, descending=True, stable=True)[:remaining]
    if remaining and not torch.isfinite(scores[selected]).all():
        raise RuntimeError("failed to satisfy bounded expert allocation")
    keep_mask.flatten()[selected] = True

    widths = keep_mask.sum(dim=1)
    if keep_mask.sum() != keep_count or (widths < lower).any() or (widths > upper).any():
        raise RuntimeError("failed to enforce bounded expert allocation")
    return keep_mask


def select_keep_mask_with_expert_widths(
    importance: torch.Tensor,
    expert_widths: torch.Tensor,
) -> torch.Tensor:
    """Select each expert's top channels under an explicit width vector."""

    if importance.ndim != 2 or min(importance.shape) == 0:
        raise ValueError("importance must have shape [experts, channels]")
    if not torch.isfinite(importance).all():
        raise ValueError("importance must contain only finite values")
    widths = torch.as_tensor(expert_widths, device=importance.device)
    if widths.shape != (importance.shape[0],):
        raise ValueError("expert_widths must contain one value per expert")
    if widths.dtype == torch.bool or widths.is_floating_point():
        raise ValueError("expert_widths must contain integers")
    widths = widths.to(torch.long)
    if (widths < 0).any() or (widths > importance.shape[1]).any():
        raise ValueError("expert_widths must lie inside the channel range")

    order = torch.argsort(importance, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        importance.shape[1], device=importance.device
    ).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    mask = ranks < widths.unsqueeze(1)
    if not torch.equal(mask.sum(dim=1), widths):
        raise RuntimeError("failed to enforce explicit expert widths")
    return mask


@torch.no_grad()
def apply_qwen3_keep_mask_(
    module: torch.nn.Module, keep_mask: torch.Tensor
) -> None:
    """Zero the matching gate/up/down vectors for every pruned micro-expert."""

    experts = module.experts
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        expert_count = experts.gate_up_proj.shape[0]
        channel_count = experts.down_proj.shape[-1]
    else:
        expert_count = len(experts)
        channel_count = experts[0].up_proj.weight.shape[0]
    if keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must have boolean dtype")
    if keep_mask.shape != (expert_count, channel_count):
        raise ValueError(
            f"keep_mask must have shape {(expert_count, channel_count)}"
        )

    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        for expert_index in range(expert_count):
            pruned = ~keep_mask[expert_index].to(experts.gate_up_proj.device)
            experts.gate_up_proj[expert_index, :channel_count][pruned] = 0
            experts.gate_up_proj[expert_index, channel_count:][pruned] = 0
            experts.down_proj[expert_index, :, pruned] = 0
        return

    for expert_index, expert in enumerate(experts):
        retained = keep_mask[expert_index].to(expert.up_proj.weight.device)
        pruned = ~retained
        expert.gate_proj.weight[pruned] = 0
        expert.up_proj.weight[pruned] = 0
        expert.down_proj.weight[:, pruned] = 0


@torch.no_grad()
def apply_qwen3_channel_coefficients_(
    module: torch.nn.Module, coefficients: torch.Tensor
) -> None:
    """Fold per-channel reconstruction coefficients into down projections."""

    experts = module.experts
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        expert_count = experts.gate_up_proj.shape[0]
        channel_count = experts.down_proj.shape[-1]
        device = experts.down_proj.device
    else:
        expert_count = len(experts)
        channel_count = experts[0].up_proj.weight.shape[0]
        device = experts[0].down_proj.weight.device
    coefficients = torch.as_tensor(
        coefficients, dtype=torch.float32, device=device
    )
    if coefficients.shape != (expert_count, channel_count):
        raise ValueError(
            f"coefficients must have shape {(expert_count, channel_count)}"
        )
    if not torch.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError("coefficients must contain finite non-negative values")

    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        experts.down_proj.mul_(coefficients[:, None, :].to(experts.down_proj.dtype))
        return
    for expert_index, expert in enumerate(experts):
        expert.down_proj.weight.mul_(
            coefficients[expert_index]
            .to(expert.down_proj.weight.dtype)
            .unsqueeze(0)
        )
