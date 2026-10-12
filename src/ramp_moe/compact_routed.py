"""Route Top-K MoE tokens through compact experts in Transformers inference."""

from __future__ import annotations

import torch
from torch import nn

from .compact_experts import CompactExperts, pack_compact_experts
from .moe_adapter import resolve_decoder_layers, resolve_moe_layer
from .ragged_kernel import ragged_expert_forward


class CompactRoutedExperts(nn.Module):
    """Drop-in replacement for packed Transformers MoE expert collections.

    The router and shared experts remain on the original model.  The inputs
    are the routed experts' original ``(hidden_states, top_k_index,
    top_k_weights)`` interface; this module owns only routed-expert weights.
    """

    def __init__(self, compact: CompactExperts):
        super().__init__()
        self.register_buffer("gate", compact.gate)
        self.register_buffer("up", compact.up)
        self.register_buffer("down", compact.down)
        self.register_buffer("offsets", compact.offsets)
        self.hidden_size = compact.hidden_size
        self.max_width = compact.max_width

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("top_k_index must have shape [tokens, top_k]")
        if top_k_weights.shape != top_k_index.shape:
            raise ValueError("top_k_weights shape must match top_k_index")
        top_k = top_k_index.shape[1]
        if top_k == 0:
            return torch.zeros_like(hidden_states)
        compact = CompactExperts(
            self.gate, self.up, self.down, self.offsets,
            self.hidden_size, self.max_width,
        )
        outputs = ragged_expert_forward(
            compact,
            hidden_states,
            top_k_index.reshape(-1).contiguous(),
            top_k_weights.reshape(-1).contiguous(),
            routes_per_token=top_k,
        )
        return outputs.reshape(hidden_states.shape[0], top_k, self.hidden_size).sum(dim=1)


@torch.no_grad()
def install_compact_routed_experts_(
    model: nn.Module, masks: list[torch.Tensor]
) -> list[tuple[int, ...]]:
    """Replace packed routed experts, returning per-layer retained widths.

    Call after loading and applying any down-projection reconstruction
    coefficients.  This operation replaces modules in place and releases the
    original dense expert weights when no other references remain.
    """

    layers = resolve_decoder_layers(model)
    if len(masks) != len(layers):
        raise ValueError("one keep mask is required for each decoder layer")
    widths_by_layer = []
    for layer, mask in zip(layers, masks):
        moe = resolve_moe_layer(layer)
        if not moe.packed:
            raise ValueError("compact routed replacement currently requires packed experts")
        compact = pack_compact_experts(moe, mask)
        widths_by_layer.append(tuple(int(x) for x in mask.sum(dim=1)))
        moe.mask_module.experts = CompactRoutedExperts(compact)
    return widths_by_layer
