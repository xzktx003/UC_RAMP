"""Model-wide rate-distortion allocation for structured MoE channels."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .selection import select_camera_keep_mask, select_keep_mask_with_expert_widths


def _validate_importance(name: str, value: torch.Tensor) -> torch.Tensor:
    importance = torch.as_tensor(value)
    if importance.ndim != 2 or min(importance.shape) == 0:
        raise ValueError(f"{name} must have shape [experts, channels]")
    if not importance.is_floating_point():
        importance = importance.float()
    if not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return importance


def lamp_relative_importance(importance: torch.Tensor) -> torch.Tensor:
    """Return LAMP's marginal distortion score for one layer.

    The input is already a non-negative channel energy, so unlike magnitude
    pruning it is not squared again.  For an ascending ordering ``q_(i)``, the
    returned score is ``q_(i) / sum_{j>=i} q_(j)``.
    """

    importance = _validate_importance("importance", importance)
    flat = importance.flatten()
    sorted_values, sorted_indices = torch.sort(flat, stable=True)
    exclusive_prefix = torch.zeros_like(sorted_values)
    if sorted_values.numel() > 1:
        exclusive_prefix[1:] = sorted_values.cumsum(dim=0)[:-1]
    surviving_energy = sorted_values.sum() - exclusive_prefix
    normalized = torch.zeros_like(sorted_values)
    positive = surviving_energy > 0
    normalized[positive] = sorted_values[positive] / surviving_energy[positive]
    result = torch.empty_like(normalized)
    result[sorted_indices] = normalized
    return result.reshape_as(importance)


def _allocation_scores(
    importance: torch.Tensor,
    *,
    score_mode: str,
) -> torch.Tensor:
    importance = _validate_importance("importance", importance)
    if score_mode == "lamp":
        return lamp_relative_importance(importance)
    if score_mode == "raw":
        return importance
    raise ValueError("score_mode must be 'lamp' or 'raw'")


def select_layer_expert_rate_distortion_masks(
    allocation_importances: Sequence[torch.Tensor],
    *,
    reference_importances: Sequence[torch.Tensor],
    global_keep_count: int,
    nominal_prune_ratio: float,
    max_width_delta: int,
    score_mode: str = "lamp",
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Allocate one exact model budget across layers and experts.

    Each layer's raw allocation energy is converted to a LAMP relative
    marginal-distortion score.  All layers then compete under one global
    budget.  Per-expert widths remain inside a trust region centered on the
    nominal anchor allocation, and final channel identities always follow the
    reference importance rather than the capacity signal.
    """

    if not allocation_importances:
        raise ValueError("allocation_importances must contain at least one layer")
    if len(allocation_importances) != len(reference_importances):
        raise ValueError("allocation and reference layers must have equal length")
    if isinstance(global_keep_count, bool) or not isinstance(global_keep_count, int):
        raise ValueError("global_keep_count must be an integer")
    if isinstance(max_width_delta, bool) or not isinstance(max_width_delta, int):
        raise ValueError("max_width_delta must be a non-negative integer")
    if max_width_delta < 0:
        raise ValueError("max_width_delta must be a non-negative integer")
    if not 0 <= nominal_prune_ratio < 1:
        raise ValueError("nominal_prune_ratio must be in [0, 1)")

    allocations = tuple(
        _validate_importance(f"allocation_importances[{index}]", value)
        for index, value in enumerate(allocation_importances)
    )
    references = tuple(
        _validate_importance(f"reference_importances[{index}]", value)
        for index, value in enumerate(reference_importances)
    )
    shape = allocations[0].shape
    device = allocations[0].device
    dtype = allocations[0].dtype
    for index, (allocation, reference) in enumerate(zip(allocations, references)):
        if allocation.shape != shape or reference.shape != shape:
            raise ValueError("all allocation and reference layers must share shape")
        if allocation.device != device or reference.device != device:
            raise ValueError("all allocation and reference layers must share device")
        if allocation.dtype != dtype or reference.dtype != dtype:
            raise ValueError("all allocation and reference layers must share dtype")

    layer_count = len(allocations)
    expert_count, channel_count = shape
    total_channels = layer_count * expert_count * channel_count
    if not 0 <= global_keep_count <= total_channels:
        raise ValueError("global_keep_count is outside the model channel range")

    lowers: list[torch.Tensor] = []
    uppers: list[torch.Tensor] = []
    mandatory_masks: list[torch.Tensor] = []
    eligible_masks: list[torch.Tensor] = []
    normalized_scores: list[torch.Tensor] = []
    reference_widths: list[torch.Tensor] = []
    for allocation, reference in zip(allocations, references):
        reference_mask = select_camera_keep_mask(
            reference,
            prune_ratio=nominal_prune_ratio,
            allocation="global",
            min_keep_per_expert=0,
        )
        widths = reference_mask.sum(dim=1)
        lower = (widths - max_width_delta).clamp_min(0)
        upper = (widths + max_width_delta).clamp_max(channel_count)
        order = torch.argsort(reference, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(channel_count, device=device).expand_as(order),
        )
        reference_widths.append(widths)
        lowers.append(lower)
        uppers.append(upper)
        mandatory_masks.append(ranks < lower.unsqueeze(1))
        eligible_masks.append(ranks < upper.unsqueeze(1))
        normalized_scores.append(
            _allocation_scores(allocation, score_mode=score_mode)
        )

    mandatory_count = sum(int(mask.sum().item()) for mask in mandatory_masks)
    eligible_count = sum(int(mask.sum().item()) for mask in eligible_masks)
    if not mandatory_count <= global_keep_count <= eligible_count:
        raise ValueError(
            "global_keep_count is infeasible under the expert width trust region"
        )

    # Interleave layers in the flattened representation.  Stable top-k then
    # distributes exact score ties across identical layers instead of always
    # favoring the earliest layer.
    score_tensor = torch.stack(normalized_scores).permute(1, 2, 0).contiguous()
    mandatory = torch.stack(mandatory_masks).permute(1, 2, 0).contiguous()
    eligible = torch.stack(eligible_masks).permute(1, 2, 0).contiguous()
    selected = mandatory.clone()
    remaining = global_keep_count - mandatory_count
    candidate_scores = score_tensor.flatten().clone()
    unavailable = (~eligible | mandatory).flatten()
    candidate_scores[unavailable] = -torch.inf
    if remaining:
        chosen = torch.argsort(
            candidate_scores, descending=True, stable=True
        )[:remaining]
        if not torch.isfinite(candidate_scores[chosen]).all():
            raise RuntimeError("failed to satisfy the model-wide keep budget")
        selected.flatten()[chosen] = True

    provisional = selected.permute(2, 0, 1).contiguous()
    masks: list[torch.Tensor] = []
    retained_widths: list[torch.Tensor] = []
    for layer_index, reference in enumerate(references):
        widths = provisional[layer_index].sum(dim=1)
        mask = select_keep_mask_with_expert_widths(reference, widths)
        if (widths < lowers[layer_index]).any() or (
            widths > uppers[layer_index]
        ).any():
            raise RuntimeError("rate-distortion allocation violated expert bounds")
        masks.append(mask)
        retained_widths.append(widths)

    retained = sum(int(mask.sum().item()) for mask in masks)
    if retained != global_keep_count:
        raise RuntimeError("rate-distortion allocation violated the exact budget")
    trace: dict[str, object] = {
        "method": (
            "layer_expert_lamp_rate_distortion"
            if score_mode == "lamp"
            else "layer_expert_raw_energy_allocation"
        ),
        "score_mode": score_mode,
        "global_keep_count": global_keep_count,
        "total_channels": total_channels,
        "nominal_prune_ratio": nominal_prune_ratio,
        "max_width_delta": max_width_delta,
        "layer_retained_channels": [
            int(mask.sum().item()) for mask in masks
        ],
        "layer_retained_ratios": [
            float(mask.float().mean().item()) for mask in masks
        ],
        "reference_expert_widths": [
            value.tolist() for value in reference_widths
        ],
        "expert_width_lower_bounds": [value.tolist() for value in lowers],
        "expert_width_upper_bounds": [value.tolist() for value in uppers],
        "expert_retained_channels": [
            value.tolist() for value in retained_widths
        ],
    }
    return tuple(masks), trace


def select_layer_expert_rate_distortion_masks_with_layer_budgets(
    allocation_importances: Sequence[torch.Tensor],
    *,
    reference_importances: Sequence[torch.Tensor],
    layer_keep_counts: Sequence[int],
    nominal_prune_ratio: float,
    max_width_delta: int,
    score_mode: str = "lamp",
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Allocate expert widths independently under fixed per-layer budgets.

    This is the controlled ablation for model-wide rate-distortion transport:
    LAMP scores and expert trust regions are unchanged, but retained channels
    cannot move between decoder layers.
    """

    if not allocation_importances:
        raise ValueError("allocation_importances must contain at least one layer")
    if len(allocation_importances) != len(reference_importances):
        raise ValueError("allocation and reference layers must have equal length")
    if len(layer_keep_counts) != len(allocation_importances):
        raise ValueError("layer_keep_counts must contain one budget per layer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in layer_keep_counts
    ):
        raise ValueError("layer_keep_counts must contain non-negative integers")

    masks: list[torch.Tensor] = []
    per_layer_traces: list[dict[str, object]] = []
    for allocation, reference, keep_count in zip(
        allocation_importances, reference_importances, layer_keep_counts
    ):
        layer_masks, layer_trace = select_layer_expert_rate_distortion_masks(
            (allocation,),
            reference_importances=(reference,),
            global_keep_count=keep_count,
            nominal_prune_ratio=nominal_prune_ratio,
            max_width_delta=max_width_delta,
            score_mode=score_mode,
        )
        masks.append(layer_masks[0])
        per_layer_traces.append(layer_trace)

    total_channels = sum(mask.numel() for mask in masks)
    trace: dict[str, object] = {
        "method": (
            "expert_lamp_rate_distortion_fixed_layer_budgets"
            if score_mode == "lamp"
            else "expert_raw_energy_fixed_layer_budgets"
        ),
        "score_mode": score_mode,
        "global_keep_count": sum(layer_keep_counts),
        "total_channels": total_channels,
        "nominal_prune_ratio": nominal_prune_ratio,
        "max_width_delta": max_width_delta,
        "layer_retained_channels": [int(mask.sum().item()) for mask in masks],
        "layer_retained_ratios": [
            float(mask.float().mean().item()) for mask in masks
        ],
        "reference_expert_widths": [
            trace["reference_expert_widths"][0] for trace in per_layer_traces
        ],
        "expert_width_lower_bounds": [
            trace["expert_width_lower_bounds"][0] for trace in per_layer_traces
        ],
        "expert_width_upper_bounds": [
            trace["expert_width_upper_bounds"][0] for trace in per_layer_traces
        ],
        "expert_retained_channels": [
            trace["expert_retained_channels"][0] for trace in per_layer_traces
        ],
    }
    return tuple(masks), trace
