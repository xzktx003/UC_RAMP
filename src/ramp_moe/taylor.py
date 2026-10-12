"""Differentiable channel gates for end-to-end mask Taylor gradients."""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class Qwen3ChannelGateController:
    """Runtime selector shared by all temporarily gated experts."""

    gates: torch.Tensor
    use_dense: bool = False

    def gate_for(
        self,
        layer_index: int,
        expert_index: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        gate = self.gates[layer_index, expert_index]
        if self.use_dense:
            return torch.ones_like(gate, dtype=dtype, device=device)
        return gate.to(dtype=dtype, device=device)


@contextmanager
def differentiable_qwen3_channel_gates(
    model: torch.nn.Module,
    gates: torch.Tensor,
) -> Iterator[Qwen3ChannelGateController]:
    """Multiply every standard Qwen3 MoE intermediate by a shared gate tensor.

    Dense weights remain untouched, so a zero-initialized (currently pruned)
    channel still receives the exact derivative of restoring its dense SwiGLU
    contribution.  The original expert methods are restored even on failure.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model must expose model.layers")
    first_experts = getattr(getattr(layers[0], "mlp", None), "experts", None)
    if first_experts is None or not isinstance(first_experts, torch.nn.ModuleList):
        raise ValueError("differentiable gates require standard ModuleList experts")
    expert_count = len(first_experts)
    if expert_count == 0:
        raise ValueError("model must contain at least one expert")
    first_expert = first_experts[0]
    required = ("gate_proj", "up_proj", "down_proj", "act_fn")
    if any(not hasattr(first_expert, name) for name in required):
        raise ValueError("Qwen3 expert is missing standard SwiGLU projections")
    channel_count = int(first_expert.up_proj.weight.shape[0])
    expected_shape = (len(layers), expert_count, channel_count)
    if gates.shape != expected_shape:
        raise ValueError(f"gates must have shape {expected_shape}")
    if not torch.is_floating_point(gates) or not torch.isfinite(gates).all():
        raise ValueError("gates must be finite floating-point values")

    controller = Qwen3ChannelGateController(gates=gates)
    restorations: list[tuple[torch.nn.Module, bool, object | None]] = []

    def make_forward(
        layer_index: int, expert_index: int
    ) -> object:
        def gated_forward(
            expert: torch.nn.Module, values: torch.Tensor
        ) -> torch.Tensor:
            hidden = expert.act_fn(expert.gate_proj(values)) * expert.up_proj(
                values
            )
            channel_gate = controller.gate_for(
                layer_index,
                expert_index,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            return expert.down_proj(hidden * channel_gate)

        return gated_forward

    try:
        for layer_index, layer in enumerate(layers):
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if not isinstance(experts, torch.nn.ModuleList) or len(experts) != expert_count:
                raise ValueError("all layers must share the standard expert layout")
            for expert_index, expert in enumerate(experts):
                if any(not hasattr(expert, name) for name in required):
                    raise ValueError("all experts must use standard SwiGLU projections")
                if int(expert.up_proj.weight.shape[0]) != channel_count:
                    raise ValueError("all experts must share the channel count")
                had_instance_forward = "forward" in expert.__dict__
                original_instance_forward = expert.__dict__.get("forward")
                restorations.append(
                    (expert, had_instance_forward, original_instance_forward)
                )
                expert.forward = types.MethodType(
                    make_forward(layer_index, expert_index), expert
                )
        yield controller
    finally:
        for expert, had_instance_forward, original in reversed(restorations):
            if had_instance_forward:
                expert.forward = original
            elif "forward" in expert.__dict__:
                del expert.__dict__["forward"]
