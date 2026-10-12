"""Deterministic multi-view consensus repair for compact MoE masks."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def splice_capability_thirds(
    code_masks: Sequence[torch.Tensor],
    general_masks: Sequence[torch.Tensor],
    math_masks: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[dict[str, int | str]]]:
    """Compose fixed code/general/math depth thirds without changing budgets."""

    layer_count = len(general_masks)
    if layer_count == 0 or layer_count % 3:
        raise ValueError("capability splice requires a layer count divisible by 3")
    if len(code_masks) != layer_count or len(math_masks) != layer_count:
        raise ValueError("all capability mask sets must have the same layer count")
    boundary = layer_count // 3
    spliced: list[torch.Tensor] = []
    summaries: list[dict[str, int | str]] = []
    for layer_index, raw_masks in enumerate(
        zip(code_masks, general_masks, math_masks)
    ):
        code, general, math = (torch.as_tensor(mask) for mask in raw_masks)
        if (
            general.dtype != torch.bool
            or general.ndim != 2
            or code.dtype != torch.bool
            or math.dtype != torch.bool
            or code.shape != general.shape
            or math.shape != general.shape
        ):
            raise ValueError("capability masks must have matching boolean layer shapes")
        retained = {int(mask.sum().item()) for mask in (code, general, math)}
        if len(retained) != 1:
            raise ValueError("capability donor layer budgets must match")
        if layer_index < boundary:
            source = "code"
            selected = code
        elif layer_index < 2 * boundary:
            source = "general"
            selected = general
        else:
            source = "math"
            selected = math
        selected = selected.clone()
        spliced.append(selected)
        summaries.append(
            {
                "layer": layer_index,
                "source": source,
                "changed_mask_bits_vs_general": int(
                    (selected != general).sum().item()
                ),
                "retained_channels": int(selected.sum().item()),
                "total_channels": selected.numel(),
            }
        )
    return spliced, summaries


def _validate_expert_masks(
    reference: torch.Tensor, donors: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    reference = torch.as_tensor(reference)
    if reference.dtype != torch.bool or reference.ndim != 1:
        raise ValueError("reference expert mask must be a one-dimensional boolean tensor")
    if len(donors) != 3:
        raise ValueError("consensus repair requires exactly three donor masks")
    normalized = tuple(torch.as_tensor(mask, device=reference.device) for mask in donors)
    if any(mask.dtype != torch.bool or mask.shape != reference.shape for mask in normalized):
        raise ValueError("donor expert masks must match the reference shape and dtype")
    return reference, normalized


def repair_expert_mask_by_consensus(
    reference: torch.Tensor,
    donors: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Swap a bounded number of channels only when a donor majority agrees.

    The reference fixes the expert width. Three independently generated donor
    masks vote on channel support. A non-reference channel must receive at
    least two votes and strictly exceed the vote count of the reference
    channel it replaces. At most one eighth of retained channels are swapped.
    """

    reference, normalized = _validate_expert_masks(reference, donors)
    repaired = reference.clone()
    keep_count = int(reference.sum().item())
    if keep_count in {0, reference.numel()}:
        return repaired
    max_swaps = max(1, keep_count // 8)
    votes = torch.stack(normalized).sum(dim=0)
    retained = torch.nonzero(reference, as_tuple=False).flatten()
    pruned = torch.nonzero(~reference, as_tuple=False).flatten()
    incoming = pruned[
        torch.argsort(votes[pruned], descending=True, stable=True)
    ]
    outgoing = retained[
        torch.argsort(votes[retained], descending=False, stable=True)
    ]
    pair_count = min(max_swaps, incoming.numel(), outgoing.numel())
    if pair_count == 0:
        return repaired
    incoming = incoming[:pair_count]
    outgoing = outgoing[:pair_count]
    accepted = (votes[incoming] >= 2) & (votes[incoming] > votes[outgoing])
    incoming = incoming[accepted]
    outgoing = outgoing[accepted]
    repaired[outgoing] = False
    repaired[incoming] = True
    if repaired.sum() != keep_count:
        raise RuntimeError("consensus repair changed the expert width")
    return repaired


def repair_tail_masks_by_consensus(
    reference_masks: Sequence[torch.Tensor],
    donor_mask_sets: Sequence[Sequence[torch.Tensor]],
) -> tuple[list[torch.Tensor], list[dict[str, int]]]:
    """Keep the first two thirds fixed and repair final-third expert channels."""

    if len(donor_mask_sets) != 3:
        raise ValueError("tail consensus requires exactly three donor mask sets")
    layer_count = len(reference_masks)
    if layer_count == 0 or layer_count % 3:
        raise ValueError("tail consensus requires a layer count divisible by 3")
    if any(len(mask_set) != layer_count for mask_set in donor_mask_sets):
        raise ValueError("every donor mask set must match the reference layer count")
    tail_start = 2 * layer_count // 3
    repaired_masks: list[torch.Tensor] = []
    summaries: list[dict[str, int]] = []
    for layer_index, reference in enumerate(reference_masks):
        reference = torch.as_tensor(reference)
        if reference.dtype != torch.bool or reference.ndim != 2:
            raise ValueError("reference layer masks must be two-dimensional boolean tensors")
        layer_donors = tuple(
            torch.as_tensor(mask_set[layer_index], device=reference.device)
            for mask_set in donor_mask_sets
        )
        if any(
            donor.dtype != torch.bool or donor.shape != reference.shape
            for donor in layer_donors
        ):
            raise ValueError("donor layer masks must match the reference shape and dtype")
        if layer_index < tail_start:
            repaired = reference.clone()
        else:
            repaired = torch.stack(
                [
                    repair_expert_mask_by_consensus(
                        reference[expert_index],
                        tuple(donor[expert_index] for donor in layer_donors),
                    )
                    for expert_index in range(reference.shape[0])
                ]
            )
        bit_changes = int((repaired != reference).sum().item())
        if bit_changes % 2:
            raise RuntimeError("consensus repair produced an unpaired bit change")
        repaired_masks.append(repaired)
        summaries.append(
            {
                "layer": layer_index,
                "swap_count": bit_changes // 2,
                "changed_experts": int(
                    (repaired != reference).any(dim=1).sum().item()
                ),
                "retained_channels": int(repaired.sum().item()),
                "total_channels": repaired.numel(),
            }
        )
    if sum(int(mask.sum()) for mask in repaired_masks) != sum(
        int(mask.sum()) for mask in reference_masks
    ):
        raise RuntimeError("tail consensus repair changed the model-wide budget")
    return repaired_masks, summaries


def splice_tail_masks(
    reference_masks: Sequence[torch.Tensor],
    donor_masks: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[dict[str, int | str]]]:
    """Use a reference prefix and the donor's complete final-third mask field."""

    layer_count = len(reference_masks)
    if layer_count == 0 or layer_count % 3:
        raise ValueError("tail splice requires a layer count divisible by 3")
    if len(donor_masks) != layer_count:
        raise ValueError("donor mask set must match the reference layer count")
    tail_start = 2 * layer_count // 3
    spliced: list[torch.Tensor] = []
    summaries: list[dict[str, int | str]] = []
    for layer_index, (raw_reference, raw_donor) in enumerate(
        zip(reference_masks, donor_masks)
    ):
        reference = torch.as_tensor(raw_reference)
        donor = torch.as_tensor(raw_donor, device=reference.device)
        if (
            reference.dtype != torch.bool
            or reference.ndim != 2
            or donor.dtype != torch.bool
            or donor.shape != reference.shape
        ):
            raise ValueError(
                "reference and donor layer masks must have matching boolean shapes"
            )
        if not torch.equal(reference.sum(dim=1), donor.sum(dim=1)):
            raise ValueError("reference and donor expert widths must match")
        source = "reference" if layer_index < tail_start else "donor"
        selected = reference.clone() if source == "reference" else donor.clone()
        spliced.append(selected)
        summaries.append(
            {
                "layer": layer_index,
                "source": source,
                "changed_mask_bits_vs_reference": int(
                    (selected != reference).sum().item()
                ),
                "retained_channels": int(selected.sum().item()),
                "total_channels": selected.numel(),
            }
        )
    return spliced, summaries
