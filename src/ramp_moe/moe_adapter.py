"""Architecture-neutral views over supported sparse MoE decoder layers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class MoELayerView:
    """Resolved router and expert storage for one sparse decoder layer."""

    router: nn.Module
    experts: nn.Module
    mask_module: nn.Module
    expert_count: int
    channel_count: int
    hidden_size: int
    packed: bool


def _resolve_path(root: object, path: tuple[str, ...]) -> object | None:
    value = root
    for name in path:
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def resolve_decoder_layers(model: object) -> Sequence[nn.Module]:
    """Return decoder layers from text-only or multimodal model wrappers."""

    paths = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
    )
    for path in paths:
        layers = _resolve_path(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            if not layers:
                raise ValueError("decoder layer collection must not be empty")
            if not all(isinstance(layer, nn.Module) for layer in layers):
                raise TypeError("decoder layers must contain torch modules")
            return layers
    raise ValueError("unable to locate decoder layers on the loaded model")


def resolve_input_embeddings(model: object) -> nn.Module:
    """Return the text token embedding across supported model wrappers."""

    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embedding = getter()
        if isinstance(embedding, nn.Module):
            return embedding
    paths = (
        ("model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
        ("language_model", "embed_tokens"),
    )
    for path in paths:
        embedding = _resolve_path(model, path)
        if isinstance(embedding, nn.Module):
            return embedding
    raise ValueError("unable to locate text input embeddings on the loaded model")


def resolve_final_norm(model: object) -> nn.Module:
    """Return the final text normalization module across model wrappers."""

    paths = (
        ("model", "norm"),
        ("model", "language_model", "norm"),
        ("language_model", "norm"),
    )
    for path in paths:
        norm = _resolve_path(model, path)
        if isinstance(norm, nn.Module):
            return norm
    raise ValueError("unable to locate final text normalization on the model")


def normalization_output_scale(norm: nn.Module) -> torch.Tensor:
    """Return the multiplicative output scale used by a supported RMSNorm."""

    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise TypeError("normalization module must expose a vector weight")
    if type(norm).__name__ == "Qwen3_5MoeRMSNorm":
        return weight + 1
    return weight


def _packed_expert_shape(experts: nn.Module) -> tuple[int, int, int] | None:
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        return None
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("packed expert projections must be rank-three tensors")
    expert_count, doubled_channels, hidden_size = gate_up.shape
    if down.shape[0] != expert_count or down.shape[1] != hidden_size:
        raise ValueError("packed gate/up and down projections are incompatible")
    channel_count = down.shape[2]
    if doubled_channels != 2 * channel_count:
        raise ValueError("packed gate/up projection must contain two channel halves")
    return expert_count, channel_count, hidden_size


def _legacy_expert_shape(experts: nn.Module) -> tuple[int, int, int]:
    if not isinstance(experts, nn.ModuleList) or not experts:
        raise TypeError("experts must use packed tensors or a non-empty ModuleList")
    first = experts[0]
    up_weight = getattr(getattr(first, "up_proj", None), "weight", None)
    gate_weight = getattr(getattr(first, "gate_proj", None), "weight", None)
    down_weight = getattr(getattr(first, "down_proj", None), "weight", None)
    if not all(
        isinstance(weight, torch.Tensor)
        for weight in (up_weight, gate_weight, down_weight)
    ):
        raise TypeError("legacy experts must expose gate/up/down projection weights")
    channel_count, hidden_size = up_weight.shape
    expected_shapes = (
        (channel_count, hidden_size),
        (hidden_size, channel_count),
    )
    for expert in experts:
        if expert.gate_proj.weight.shape != expected_shapes[0]:
            raise ValueError("legacy expert gate projections have inconsistent shapes")
        if expert.up_proj.weight.shape != expected_shapes[0]:
            raise ValueError("legacy expert up projections have inconsistent shapes")
        if expert.down_proj.weight.shape != expected_shapes[1]:
            raise ValueError("legacy expert down projections have inconsistent shapes")
    return len(experts), channel_count, hidden_size


def resolve_moe_layer(layer: nn.Module) -> MoELayerView:
    """Resolve Qwen-style nested or Gemma-style parallel sparse experts."""

    if hasattr(layer, "experts") and (
        hasattr(layer, "router") or hasattr(layer, "gate")
    ):
        mask_module = layer
        router = getattr(layer, "router", getattr(layer, "gate", None))
        experts = layer.experts
    else:
        block = getattr(layer, "mlp", None)
        if not isinstance(block, nn.Module) or not hasattr(block, "experts"):
            raise ValueError("decoder layer does not expose a supported sparse MoE block")
        router = getattr(block, "gate", getattr(block, "router", None))
        if not isinstance(router, nn.Module):
            raise ValueError("sparse MoE block does not expose a router module")
        mask_module = block
        experts = block.experts
    if not isinstance(router, nn.Module) or not isinstance(experts, nn.Module):
        raise TypeError("router and experts must be torch modules")

    packed_shape = _packed_expert_shape(experts)
    if packed_shape is None:
        expert_count, channel_count, hidden_size = _legacy_expert_shape(experts)
        packed = False
    else:
        expert_count, channel_count, hidden_size = packed_shape
        packed = True
    return MoELayerView(
        router=router,
        experts=experts,
        mask_module=mask_module,
        expert_count=expert_count,
        channel_count=channel_count,
        hidden_size=hidden_size,
        packed=packed,
    )


def exact_layer_retained_budgets(
    schedule: str,
    *,
    layer_count: int,
    expert_count: int,
    channel_count: int,
) -> tuple[int, ...]:
    """Allocate deterministic per-layer keeps with exact model-wide 50% sparsity.

    ``tail_protected`` preserves the established 7/16 versus 10/16 depth
    preference and normalizes it when the number of layers is not divisible by
    three. Integer remainders are assigned by largest fractional remainder and
    then layer index, so no architecture-specific ratio search is introduced.
    """

    if layer_count <= 0 or expert_count <= 0 or channel_count <= 0:
        raise ValueError("layer, expert, and channel counts must be positive")
    layer_capacity = expert_count * channel_count
    global_capacity = layer_count * layer_capacity
    if global_capacity % 2:
        raise ValueError("exact 50% retention requires an even global capacity")

    if schedule == "uniform":
        raw_weights = (Fraction(1, 2),) * layer_count
    elif schedule == "tail_protected":
        tail_start = 2 * layer_count // 3
        raw_weights = tuple(
            Fraction(7, 16) if index < tail_start else Fraction(10, 16)
            for index in range(layer_count)
        )
    else:
        raise ValueError(f"unsupported layer budget schedule: {schedule}")

    target = global_capacity // 2
    raw_total = sum(raw_weights, Fraction())
    exact = tuple(
        Fraction(target) * weight / raw_total for weight in raw_weights
    )
    budgets = [value.numerator // value.denominator for value in exact]
    remaining = target - sum(budgets)
    remainder_order = sorted(
        range(layer_count),
        key=lambda index: (
            -(exact[index] - budgets[index]),
            index,
        ),
    )
    for index in remainder_order[:remaining]:
        budgets[index] += 1

    if sum(budgets) != target:
        raise RuntimeError("failed to allocate the exact model-wide 50% budget")
    if any(budget < 0 or budget > layer_capacity for budget in budgets):
        raise RuntimeError("normalized layer budget lies outside layer capacity")
    return tuple(budgets)
