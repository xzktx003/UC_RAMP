"""Independent, numerically stable implementation of CAMERA-P importance."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    top_k: int,
    alpha: float,
    token_chunk_size: int,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    if hidden_states.shape[0] == 0:
        raise ValueError("hidden_states must contain at least one token")
    if router_weight.ndim != 2:
        raise ValueError("router_weight must have shape [experts, hidden_size]")
    if gate_weights.ndim != 3 or up_weights.ndim != 3 or down_weights.ndim != 3:
        raise ValueError("expert projection weights must be rank-three tensors")
    if gate_weights.shape != up_weights.shape:
        raise ValueError("gate and up expert weights must have identical shapes")

    expert_count, channel_count, hidden_size = gate_weights.shape
    if router_weight.shape != (expert_count, hidden_size):
        raise ValueError("router and expert weight shapes are inconsistent")
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hidden state size does not match model weights")
    if down_weights.shape != (expert_count, hidden_size, channel_count):
        raise ValueError("down expert weight shape is inconsistent")
    if not 1 <= top_k <= expert_count:
        raise ValueError(f"top_k must be in [1, {expert_count}]")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")

    devices = {
        hidden_states.device,
        router_weight.device,
        gate_weights.device,
        up_weights.device,
        down_weights.device,
    }
    if len(devices) != 1:
        raise ValueError("all CAMERA inputs must be on the same device")


@torch.no_grad()
def camera_microexpert_importance(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    *,
    top_k: int,
    alpha: float = 0.9,
    token_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute CAMERA energy for every ``[expert, channel]``.

    This is equivalent to evaluating all experts and multiplying non-routed
    activations by zero, but it evaluates only routed token/expert pairs. The
    squared-L2 and max statistics are accumulated in FP32.
    """

    _validate_inputs(
        hidden_states,
        router_weight,
        gate_weights,
        up_weights,
        down_weights,
        top_k,
        alpha,
        token_chunk_size,
    )
    expert_count, channel_count, _ = gate_weights.shape
    l2_square = torch.zeros(
        (expert_count, channel_count), device=hidden_states.device, dtype=torch.float32
    )
    linf_square = torch.zeros_like(l2_square)

    for start in range(0, hidden_states.shape[0], token_chunk_size):
        chunk = hidden_states[start : start + token_chunk_size]
        router_logits = chunk @ router_weight.T
        router_probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_values, top_indices = torch.topk(
            router_probabilities, k=top_k, dim=-1
        )
        top_scores = top_values / top_values.sum(dim=-1, keepdim=True)

        for expert_tensor in torch.unique(top_indices):
            expert_index = int(expert_tensor.item())
            positions = torch.nonzero(
                top_indices == expert_index, as_tuple=False
            )
            token_rows = positions[:, 0]
            score_slots = positions[:, 1]
            expert_inputs = chunk[token_rows]
            router_scores = top_scores[token_rows, score_slots]

            intermediate = F.silu(
                expert_inputs @ gate_weights[expert_index].T
            )
            intermediate *= expert_inputs @ up_weights[expert_index].T
            routed = intermediate.float() * router_scores.unsqueeze(1)
            squared = routed.square()
            l2_square[expert_index] += squared.sum(dim=0)
            linf_square[expert_index] = torch.maximum(
                linf_square[expert_index], squared.amax(dim=0)
            )

    activation_energy = (1 - alpha) * l2_square + alpha * linf_square
    basis_energy = down_weights.float().square().sum(dim=1)
    return activation_energy * basis_energy
