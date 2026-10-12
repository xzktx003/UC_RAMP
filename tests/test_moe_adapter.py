from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ramp_moe.moe_adapter import (
    exact_layer_retained_budgets,
    normalization_output_scale,
    resolve_decoder_layers,
    resolve_final_norm,
    resolve_moe_layer,
)


class _PackedExperts(nn.Module):
    def __init__(self, expert_count: int = 4, channel_count: int = 6) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.ones(expert_count, 2 * channel_count, 5)
        )
        self.down_proj = nn.Parameter(
            torch.ones(expert_count, 5, channel_count)
        )


class _LegacyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(5, 6, bias=False)
        self.up_proj = nn.Linear(5, 6, bias=False)
        self.down_proj = nn.Linear(6, 5, bias=False)


def test_resolve_decoder_layers_supports_text_and_multimodal_wrappers() -> None:
    text_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    text_model = SimpleNamespace(model=SimpleNamespace(layers=text_layers))
    assert resolve_decoder_layers(text_model) is text_layers

    language_layers = nn.ModuleList([nn.Identity()])
    multimodal_model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=language_layers)
        )
    )
    assert resolve_decoder_layers(multimodal_model) is language_layers


def test_resolve_qwen_packed_moe_layer() -> None:
    router = nn.Linear(5, 4, bias=False)
    experts = _PackedExperts()
    block = nn.Module()
    block.gate = router
    block.experts = experts
    layer = nn.Module()
    layer.mlp = block

    view = resolve_moe_layer(layer)

    assert view.router is router
    assert view.experts is experts
    assert view.mask_module is block
    assert view.expert_count == 4
    assert view.channel_count == 6
    assert view.hidden_size == 5
    assert view.packed is True

    direct_view = resolve_moe_layer(block)
    assert direct_view.mask_module is block
    assert direct_view.router is router


def test_resolve_gemma_parallel_moe_layer() -> None:
    router = nn.Linear(5, 4, bias=False)
    experts = _PackedExperts()
    layer = nn.Module()
    layer.router = router
    layer.experts = experts

    view = resolve_moe_layer(layer)

    assert view.router is router
    assert view.experts is experts
    assert view.mask_module is layer
    assert view.expert_count == 4
    assert view.channel_count == 6
    assert view.hidden_size == 5
    assert view.packed is True


def test_resolve_legacy_qwen_module_list_experts() -> None:
    block = nn.Module()
    block.gate = nn.Linear(5, 2, bias=False)
    block.experts = nn.ModuleList([_LegacyExpert(), _LegacyExpert()])
    layer = nn.Module()
    layer.mlp = block

    view = resolve_moe_layer(layer)

    assert view.expert_count == 2
    assert view.channel_count == 6
    assert view.hidden_size == 5
    assert view.packed is False


@pytest.mark.parametrize("layer_count", [30, 40, 48])
def test_tail_protected_budget_is_exact_for_target_architectures(
    layer_count: int,
) -> None:
    layer_capacity = 128 * 704
    budgets = exact_layer_retained_budgets(
        "tail_protected",
        layer_count=layer_count,
        expert_count=128,
        channel_count=704,
    )

    assert len(budgets) == layer_count
    assert sum(budgets) == layer_count * layer_capacity // 2
    tail_start = 2 * layer_count // 3
    assert min(budgets[tail_start:]) > max(budgets[:tail_start])
    assert max(budgets[:tail_start]) - min(budgets[:tail_start]) <= 1
    assert max(budgets[tail_start:]) - min(budgets[tail_start:]) <= 1


def test_tail_protected_budget_reproduces_original_48_layer_schedule() -> None:
    budgets = exact_layer_retained_budgets(
        "tail_protected",
        layer_count=48,
        expert_count=2,
        channel_count=8,
    )

    assert budgets == (7,) * 32 + (10,) * 16


def test_uniform_budget_rejects_impossible_odd_global_total() -> None:
    with pytest.raises(ValueError, match="even"):
        exact_layer_retained_budgets(
            "uniform",
            layer_count=3,
            expert_count=1,
            channel_count=1,
        )


def test_qwen35_zero_centered_rms_weight_maps_to_unit_output_scale() -> None:
    class Qwen3_5MoeRMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

    qwen_norm = Qwen3_5MoeRMSNorm()
    ordinary_norm = nn.LayerNorm(4)

    torch.testing.assert_close(
        normalization_output_scale(qwen_norm), torch.ones(4)
    )
    assert normalization_output_scale(ordinary_norm) is ordinary_norm.weight


def test_resolve_final_norm_supports_multimodal_wrapper() -> None:
    norm = nn.LayerNorm(4)
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(norm=norm))
    )
    assert resolve_final_norm(model) is norm
