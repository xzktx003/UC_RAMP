#!/usr/bin/env python3
"""Reload an exported sparse MoE model and audit joint channel zeros."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ramp_moe.checkpoints import REGISTERED_PRUNE_RATIOS

from ramp_moe.moe_adapter import resolve_decoder_layers, resolve_moe_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _packed_projection_support(experts: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    channel_count = experts.down_proj.shape[-1]
    gate = experts.gate_up_proj[:, :channel_count].ne(0).any(dim=-1).cpu()
    up = experts.gate_up_proj[:, channel_count:].ne(0).any(dim=-1).cpu()
    down = experts.down_proj.ne(0).any(dim=1).cpu()
    return gate, up, down


def _legacy_projection_support(experts: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    gate = torch.stack(
        [expert.gate_proj.weight.ne(0).any(dim=1).cpu() for expert in experts]
    )
    up = torch.stack(
        [expert.up_proj.weight.ne(0).any(dim=1).cpu() for expert in experts]
    )
    down = torch.stack(
        [expert.down_proj.weight.ne(0).any(dim=0).cpu() for expert in experts]
    )
    return gate, up, down


@torch.no_grad()
def inspect_joint_zero_channels(
    model: torch.nn.Module, masks: list[torch.Tensor]
) -> dict[str, Any]:
    """Count channels that violate a checkpoint's joint-zero invariant."""

    layers = resolve_decoder_layers(model)
    if len(masks) != len(layers):
        raise ValueError("mask checkpoint layer count does not match the model")
    layer_summaries: list[dict[str, Any]] = []
    total_pruned = 0
    total_violations = 0
    total_retained_joint_zero = 0
    packed_layers = 0
    for layer_index, (layer, raw_mask) in enumerate(zip(layers, masks)):
        moe = resolve_moe_layer(layer)
        mask = torch.as_tensor(raw_mask, dtype=torch.bool, device="cpu")
        expected_shape = (moe.expert_count, moe.channel_count)
        if mask.shape != expected_shape:
            raise ValueError(
                f"layer {layer_index} mask has shape {tuple(mask.shape)}, "
                f"expected {expected_shape}"
            )
        support = (
            _packed_projection_support(moe.experts)
            if moe.packed
            else _legacy_projection_support(moe.experts)
        )
        packed_layers += int(moe.packed)
        jointly_nonzero = support[0] | support[1] | support[2]
        pruned = ~mask
        violations = pruned & jointly_nonzero
        retained_joint_zero = mask & ~jointly_nonzero
        layer_summary = {
            "layer_index": layer_index,
            "packed": moe.packed,
            "expert_count": moe.expert_count,
            "channel_count": moe.channel_count,
            "retained_channel_count": int(mask.sum().item()),
            "pruned_channel_count": int(pruned.sum().item()),
            "pruned_channel_violations": int(violations.sum().item()),
            "retained_joint_zero_channels": int(retained_joint_zero.sum().item()),
        }
        layer_summaries.append(layer_summary)
        total_pruned += layer_summary["pruned_channel_count"]
        total_violations += layer_summary["pruned_channel_violations"]
        total_retained_joint_zero += layer_summary["retained_joint_zero_channels"]

    return {
        "status": "complete" if total_violations == 0 else "failed",
        "layer_count": len(layers),
        "packed_layer_count": packed_layers,
        "pruned_channel_count": total_pruned,
        "pruned_channel_violations": total_violations,
        "retained_joint_zero_channels": total_retained_joint_zero,
        "layers": layer_summaries,
    }


def _load_checkpoint(
    path: Path, *, expected_model: str, expected_revision: str
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported mask checkpoint schema")
    if checkpoint.get("model") != expected_model:
        raise ValueError("mask checkpoint model does not match --model")
    if checkpoint.get("model_revision") != expected_revision:
        raise ValueError("mask checkpoint revision does not match --revision")
    if checkpoint.get("prune_ratio") not in REGISTERED_PRUNE_RATIOS:
        raise ValueError(
            "only the registered 25% or 50% pruning checkpoints are accepted"
        )
    masks = checkpoint.get("masks")
    if not isinstance(masks, list) or not all(
        isinstance(mask, torch.Tensor) and mask.dtype == torch.bool for mask in masks
    ):
        raise ValueError("checkpoint masks must be a list of boolean tensors")
    return checkpoint


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    model_source = args.model_source.resolve()
    checkpoint_path = args.mask_checkpoint.resolve()
    checkpoint = _load_checkpoint(
        checkpoint_path,
        expected_model=args.model,
        expected_revision=args.revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    audit = inspect_joint_zero_channels(model, checkpoint["masks"])
    result = {
        "schema_version": 1,
        "experiment_type": "masked_model_export_verification",
        "model": args.model,
        "model_revision": args.revision,
        "model_source": str(model_source),
        "mask_checkpoint": str(checkpoint_path),
        "mask_checkpoint_sha256": _sha256(checkpoint_path),
        "method": checkpoint["method"],
        "allocation": checkpoint["allocation"],
        "prune_ratio": checkpoint["prune_ratio"],
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "command": [sys.executable, *sys.argv],
        **audit,
    }
    _json_atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
