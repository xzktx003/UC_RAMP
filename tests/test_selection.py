import torch
import pytest

from ramp_moe.selection import (
    apply_qwen3_channel_coefficients_,
    apply_qwen3_keep_mask_,
    select_bounded_camera_keep_mask,
    select_camera_keep_mask,
    select_keep_mask_with_expert_widths,
)
from test_qwen3_stats import _module


def test_camera_global_budget_keeps_exactly_half() -> None:
    importance = torch.tensor([[9.0, 1.0, 8.0], [2.0, 7.0, 3.0]])

    mask = select_camera_keep_mask(
        importance, prune_ratio=0.5, allocation="global"
    )

    assert mask.tolist() == [[True, False, True], [False, True, False]]
    assert mask.sum().item() == 3


def test_bounded_camera_allocation_limits_expert_width_migration() -> None:
    reference = torch.tensor(
        [[12.0, 11.0, 10.0, 9.0], [8.0, 7.0, 6.0, 5.0]]
    )
    target = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [20.0, 19.0, 18.0, 17.0]]
    )

    mask = select_bounded_camera_keep_mask(
        target,
        reference_importance=reference,
        prune_ratio=0.5,
        max_width_delta=1,
    )

    assert mask.sum().item() == 4
    assert mask.sum(dim=1).tolist() == [3, 1]
    assert mask[0, 3]
    assert mask[1, 0]


def test_bounded_camera_allocation_rejects_infeasible_delta() -> None:
    importance = torch.ones(2, 4)

    with pytest.raises(ValueError, match="max_width_delta"):
        select_bounded_camera_keep_mask(
            importance,
            reference_importance=importance,
            max_width_delta=-1,
        )


def test_explicit_expert_widths_reselect_channels_by_independent_importance() -> None:
    importance = torch.tensor(
        [[1.0, 2.0, 9.0, 8.0], [7.0, 6.0, 5.0, 4.0]]
    )

    mask = select_keep_mask_with_expert_widths(
        importance, torch.tensor([3, 1])
    )

    assert mask.tolist() == [
        [False, True, True, True],
        [True, False, False, False],
    ]


def test_camera_uniform_budget_keeps_half_per_expert() -> None:
    importance = torch.tensor([[9.0, 1.0, 8.0, 2.0], [2.0, 7.0, 3.0, 6.0]])

    mask = select_camera_keep_mask(
        importance, prune_ratio=0.5, allocation="uniform"
    )

    assert mask.tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]


def test_camera_global_floor_preserves_every_expert_and_exact_budget() -> None:
    importance = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0, 3.0],
            [2.0, 1.0, 0.0, -1.0],
        ]
    )

    mask = select_camera_keep_mask(
        importance,
        prune_ratio=0.5,
        allocation="global",
        min_keep_per_expert=1,
    )

    assert mask.sum().item() == 6
    assert mask.sum(dim=1).tolist() == [4, 1, 1]
    assert mask[2, 0]


def test_camera_global_floor_rejects_infeasible_budget() -> None:
    importance = torch.ones(3, 2)

    with pytest.raises(ValueError, match="minimum expert budget"):
        select_camera_keep_mask(
            importance,
            prune_ratio=0.5,
            allocation="global",
            min_keep_per_expert=2,
        )


def test_camera_uniform_rejects_global_floor_option() -> None:
    with pytest.raises(ValueError, match="only supported for global"):
        select_camera_keep_mask(
            torch.ones(2, 4),
            prune_ratio=0.5,
            allocation="uniform",
            min_keep_per_expert=1,
        )


def test_joint_qwen3_mask_zeros_all_three_projection_vectors() -> None:
    module = _module()
    original_router = module.gate.weight.detach().clone()
    keep_mask = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )

    apply_qwen3_keep_mask_(module, keep_mask)

    torch.testing.assert_close(module.gate.weight, original_router)
    for expert_index, expert in enumerate(module.experts):
        pruned = ~keep_mask[expert_index]
        assert torch.count_nonzero(expert.gate_proj.weight[pruned]) == 0
        assert torch.count_nonzero(expert.up_proj.weight[pruned]) == 0
        assert torch.count_nonzero(expert.down_proj.weight[:, pruned]) == 0


def test_qwen3_channel_coefficients_only_scale_down_projection() -> None:
    module = _module()
    original_gate = [expert.gate_proj.weight.detach().clone() for expert in module.experts]
    original_up = [expert.up_proj.weight.detach().clone() for expert in module.experts]
    original_down = [expert.down_proj.weight.detach().clone() for expert in module.experts]
    coefficients = torch.tensor([[2.0, 0.5, 1.0], [1.5, 1.0, 0.25]])

    apply_qwen3_channel_coefficients_(module, coefficients)

    for index, expert in enumerate(module.experts):
        torch.testing.assert_close(expert.gate_proj.weight, original_gate[index])
        torch.testing.assert_close(expert.up_proj.weight, original_up[index])
        torch.testing.assert_close(
            expert.down_proj.weight,
            original_down[index] * coefficients[index].unsqueeze(0),
        )
