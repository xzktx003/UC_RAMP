"""Validation and application of compact pruning-mask checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .moe_adapter import resolve_decoder_layers, resolve_moe_layer
from .selection import (
    apply_qwen3_channel_coefficients_,
    apply_qwen3_keep_mask_,
)


REGISTERED_PRUNE_RATIOS = frozenset({0.25, 0.5})


@torch.no_grad()
def apply_mask_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_model: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Load, validate, and apply a pinned 25% or 50% mask checkpoint."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported mask checkpoint schema")
    if checkpoint.get("model") != expected_model:
        raise ValueError("mask checkpoint model does not match the loaded model")
    if checkpoint.get("model_revision") != expected_revision:
        raise ValueError("mask checkpoint revision does not match the loaded model")
    if checkpoint.get("prune_ratio") not in REGISTERED_PRUNE_RATIOS:
        raise ValueError(
            "only the registered 25% or 50% pruning checkpoints are accepted"
        )
    masks = checkpoint.get("masks")
    down_coefficients = checkpoint.get("down_coefficients")
    layers = resolve_decoder_layers(model)
    if not isinstance(masks, list) or len(masks) != len(layers):
        raise ValueError("mask checkpoint layer count does not match the model")
    if down_coefficients is not None and (
        not isinstance(down_coefficients, list)
        or len(down_coefficients) != len(layers)
    ):
        raise ValueError(
            "coefficient checkpoint layer count does not match the model"
        )

    for layer_index, (layer, mask) in enumerate(zip(layers, masks)):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"layer {layer_index} mask must be a boolean tensor")
        moe = resolve_moe_layer(layer)
        apply_qwen3_keep_mask_(moe.mask_module, mask)
        if down_coefficients is not None:
            coefficients = down_coefficients[layer_index]
            if not isinstance(coefficients, torch.Tensor):
                raise ValueError(
                    f"layer {layer_index} coefficients must be a tensor"
                )
            apply_qwen3_channel_coefficients_(moe.mask_module, coefficients)

    return {
        key: value
        for key, value in checkpoint.items()
        if key not in {"masks", "down_coefficients"}
    }
