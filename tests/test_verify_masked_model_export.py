from __future__ import annotations

import torch
import pytest
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from ramp_moe.moe_adapter import resolve_decoder_layers, resolve_moe_layer
from ramp_moe.selection import apply_qwen3_keep_mask_
from scripts.verify_masked_model_export import _load_checkpoint, inspect_joint_zero_channels


def _tiny_model() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    return Qwen3MoeForCausalLM(config).eval()


def test_inspect_joint_zero_channels_accepts_applied_mask() -> None:
    model = _tiny_model()
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    layer = resolve_decoder_layers(model)[0]
    apply_qwen3_keep_mask_(resolve_moe_layer(layer).mask_module, mask)

    result = inspect_joint_zero_channels(model, [mask])

    assert result["status"] == "complete"
    assert result["layer_count"] == 1
    assert result["pruned_channel_count"] == 8
    assert result["pruned_channel_violations"] == 0
    assert result["retained_joint_zero_channels"] == 0


def test_inspect_joint_zero_channels_reports_projection_violation() -> None:
    model = _tiny_model()
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    layer = resolve_decoder_layers(model)[0]
    moe = resolve_moe_layer(layer)
    apply_qwen3_keep_mask_(moe.mask_module, mask)
    experts = moe.experts
    if moe.packed:
        experts.gate_up_proj.data[0, 2, 0] = 1
    else:
        experts[0].gate_proj.weight.data[2, 0] = 1

    result = inspect_joint_zero_channels(model, [mask])

    assert result["status"] == "failed"
    assert result["pruned_channel_violations"] == 1


@pytest.mark.parametrize("prune_ratio", [0.25, 0.5])
def test_load_checkpoint_accepts_registered_pruning_endpoints(
    tmp_path, prune_ratio: float
) -> None:
    checkpoint_path = tmp_path / "mask.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model",
            "model_revision": "revision",
            "prune_ratio": prune_ratio,
            "masks": [torch.ones((1, 1), dtype=torch.bool)],
        },
        checkpoint_path,
    )

    checkpoint = _load_checkpoint(
        checkpoint_path,
        expected_model="model",
        expected_revision="revision",
    )

    assert checkpoint["prune_ratio"] == prune_ratio


def test_load_checkpoint_rejects_unregistered_pruning_endpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "mask.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model",
            "model_revision": "revision",
            "prune_ratio": 0.3,
            "masks": [torch.ones((1, 1), dtype=torch.bool)],
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="registered 25% or 50%"):
        _load_checkpoint(
            checkpoint_path,
            expected_model="model",
            expected_revision="revision",
        )
