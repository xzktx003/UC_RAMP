import pytest
import torch

from ramp_moe.rate_distortion import (
    lamp_relative_importance,
    select_layer_expert_rate_distortion_masks,
    select_layer_expert_rate_distortion_masks_with_layer_budgets,
)


def test_lamp_relative_importance_matches_surviving_energy_formula() -> None:
    importance = torch.tensor([[1.0, 2.0, 4.0]])

    normalized = lamp_relative_importance(importance)

    assert torch.allclose(
        normalized,
        torch.tensor([[1.0 / 7.0, 2.0 / 6.0, 1.0]]),
    )


def test_rate_distortion_allocation_changes_layer_and_expert_widths() -> None:
    reference = (
        torch.tensor([[100.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]]),
        torch.tensor([[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]),
    )
    allocation = (
        torch.tensor([[100.0, 0.3, 0.2, 0.1], [8.0, 0.7, 0.6, 0.5]]),
        torch.tensor([[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]),
    )

    masks, trace = select_layer_expert_rate_distortion_masks(
        allocation,
        reference_importances=reference,
        global_keep_count=8,
        nominal_prune_ratio=0.5,
        max_width_delta=2,
    )

    assert sum(int(mask.sum()) for mask in masks) == 8
    assert trace["layer_retained_channels"][0] != trace[
        "layer_retained_channels"
    ][1]
    assert any(
        len(set(widths)) > 1 for widths in trace["expert_retained_channels"]
    )
    for mask, anchor, widths in zip(
        masks, reference, trace["expert_retained_channels"]
    ):
        for expert, width in enumerate(widths):
            expected = torch.topk(anchor[expert], width).indices
            assert mask[expert, expected].all()


def test_rate_distortion_ties_are_balanced_across_identical_layers() -> None:
    layer = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    masks, trace = select_layer_expert_rate_distortion_masks(
        (layer, layer.clone()),
        reference_importances=(layer, layer.clone()),
        global_keep_count=4,
        nominal_prune_ratio=0.5,
        max_width_delta=2,
    )

    assert [int(mask.sum()) for mask in masks] == [2, 2]
    assert trace["layer_retained_channels"] == [2, 2]


def test_raw_energy_ablation_removes_lamp_layer_scale_normalization() -> None:
    allocation = (
        torch.tensor([[100.0, 2.0]]),
        torch.tensor([[1.0, 0.9]]),
    )
    reference = tuple(value.clone() for value in allocation)

    lamp_masks, lamp_trace = select_layer_expert_rate_distortion_masks(
        allocation,
        reference_importances=reference,
        global_keep_count=2,
        nominal_prune_ratio=0.5,
        max_width_delta=1,
        score_mode="lamp",
    )
    raw_masks, raw_trace = select_layer_expert_rate_distortion_masks(
        allocation,
        reference_importances=reference,
        global_keep_count=2,
        nominal_prune_ratio=0.5,
        max_width_delta=1,
        score_mode="raw",
    )

    assert [int(mask.sum()) for mask in lamp_masks] == [1, 1]
    assert [int(mask.sum()) for mask in raw_masks] == [2, 0]
    assert lamp_trace["score_mode"] == "lamp"
    assert raw_trace["score_mode"] == "raw"


def test_zero_width_delta_freezes_every_expert_to_anchor_width() -> None:
    allocation = (
        torch.tensor([[100.0, 2.0, 1.0, 0.5], [9.0, 8.0, 7.0, 6.0]]),
        torch.tensor([[5.0, 4.0, 3.0, 2.0], [1.0, 0.5, 0.2, 0.1]]),
    )
    reference = tuple(value.clone() for value in allocation)

    masks, trace = select_layer_expert_rate_distortion_masks(
        allocation,
        reference_importances=reference,
        global_keep_count=8,
        nominal_prune_ratio=0.5,
        max_width_delta=0,
    )

    assert sum(int(mask.sum()) for mask in masks) == 8
    assert trace["max_width_delta"] == 0
    assert trace["expert_retained_channels"] == trace["reference_expert_widths"]
    assert trace["expert_width_lower_bounds"] == trace[
        "expert_width_upper_bounds"
    ]


def test_rate_distortion_allocation_enforces_anchor_width_bounds() -> None:
    layers = tuple(
        torch.tensor(
            [
                [16.0, 8.0, 4.0, 2.0],
                [8.0, 4.0, 2.0, 1.0],
            ]
        )
        for _ in range(2)
    )

    masks, trace = select_layer_expert_rate_distortion_masks(
        layers,
        reference_importances=layers,
        global_keep_count=8,
        nominal_prune_ratio=0.5,
        max_width_delta=1,
    )

    assert sum(int(mask.sum()) for mask in masks) == 8
    for widths, lower, upper in zip(
        trace["expert_retained_channels"],
        trace["expert_width_lower_bounds"],
        trace["expert_width_upper_bounds"],
    ):
        assert all(lo <= width <= hi for width, lo, hi in zip(widths, lower, upper))


def test_rate_distortion_allocation_rejects_infeasible_budget() -> None:
    layer = torch.ones((2, 4))

    with pytest.raises(ValueError, match="infeasible"):
        select_layer_expert_rate_distortion_masks(
            (layer,),
            reference_importances=(layer,),
            global_keep_count=0,
            nominal_prune_ratio=0.5,
            max_width_delta=0,
        )


def test_layer_budget_ablation_prevents_cross_layer_budget_transport() -> None:
    reference = (
        torch.tensor([[100.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]]),
        torch.tensor([[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]),
    )
    allocation = (
        torch.tensor([[100.0, 0.3, 0.2, 0.1], [8.0, 0.7, 0.6, 0.5]]),
        torch.tensor([[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]),
    )

    masks, trace = select_layer_expert_rate_distortion_masks_with_layer_budgets(
        allocation,
        reference_importances=reference,
        layer_keep_counts=(4, 4),
        nominal_prune_ratio=0.5,
        max_width_delta=2,
    )

    assert [int(mask.sum()) for mask in masks] == [4, 4]
    assert trace["method"] == "expert_lamp_rate_distortion_fixed_layer_budgets"
    assert trace["layer_retained_channels"] == [4, 4]
    assert sum(trace["layer_retained_channels"]) == 8
    assert any(
        len(set(widths)) > 1 for widths in trace["expert_retained_channels"]
    )
