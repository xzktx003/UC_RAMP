#!/usr/bin/env python3
"""Portable Boolean mask artifacts; never package model weights."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


METADATA_KEYS = (
    "schema_version", "model", "model_revision", "method", "allocation",
    "prune_ratio", "camera_alpha", "anchor_general_weight", "trust_radius_divisor",
    "interaction_strength", "max_swaps", "calibration_aggregation",
    "layer_budget_schedule", "layer_budgets", "rate_distortion_allocation",
    "fixed_stage1_checkpoint_sha256",
)


def encode_checkpoint(checkpoint: dict) -> dict:
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if checkpoint.get("down_coefficients") is not None:
        raise ValueError("release masks must not contain weight recovery coefficients")
    masks = checkpoint.get("masks")
    if not isinstance(masks, list) or not masks:
        raise ValueError("missing masks")
    records = []
    for mask in masks:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("expected Boolean [expert, channel] masks")
        raw = np.packbits(mask.cpu().numpy().reshape(-1), bitorder="little").tobytes()
        records.append({"shape": list(mask.shape), "sha256": hashlib.sha256(raw).hexdigest(),
                        "packed_bits_base64": base64.b64encode(raw).decode("ascii")})
    return {"format": "uc-ramp-boolean-mask-v1", "bitorder": "little",
            "metadata": {key: checkpoint[key] for key in METADATA_KEYS if key in checkpoint},
            "layers": records}


def decode_artifact(artifact: dict) -> dict:
    if artifact.get("format") != "uc-ramp-boolean-mask-v1" or artifact.get("bitorder") != "little":
        raise ValueError("unsupported mask artifact")
    metadata = artifact["metadata"]
    if set(metadata) - set(METADATA_KEYS) or metadata.get("schema_version") != 1:
        raise ValueError("unexpected checkpoint metadata")
    records = artifact["layers"]
    if not records:
        raise ValueError("missing mask layers")
    masks = []
    for record in records:
        shape = record["shape"]
        if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("invalid mask shape")
        raw = base64.b64decode(record["packed_bits_base64"], validate=True)
        count = shape[0] * shape[1]
        if len(raw) != (count + 7) // 8 or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("mask checksum or size mismatch")
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        if np.any(bits[count:]):
            raise ValueError("nonzero mask padding bits")
        masks.append(torch.from_numpy(bits[:count].reshape(shape).astype(np.bool_)))
    total = sum(mask.numel() for mask in masks)
    kept = sum(int(mask.sum()) for mask in masks)
    ratio = metadata.get("prune_ratio")
    if ratio not in (0.25, 0.5) or kept != round(total * (1 - ratio)):
        raise ValueError("mask violates the exact channel budget")
    if "layer_budgets" in metadata and metadata["layer_budgets"] != [int(m.sum()) for m in masks]:
        raise ValueError("per-layer channel budget mismatch")
    return {**metadata, "masks": masks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "unpack", "verify"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action != "verify" and args.output is None:
        parser.error("output is required for pack/unpack")
    if args.output and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.action == "pack":
        artifact = encode_checkpoint(torch.load(args.input, map_location="cpu", weights_only=True))
        decode_artifact(artifact)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    else:
        checkpoint = decode_artifact(json.loads(args.input.read_text(encoding="utf-8")))
        if args.action == "unpack":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output)
        print(json.dumps({"status": "verified", "layers": len(checkpoint["masks"]),
                          "kept_channels": sum(int(m.sum()) for m in checkpoint["masks"])}))


if __name__ == "__main__":
    main()
