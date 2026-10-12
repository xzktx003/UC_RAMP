#!/usr/bin/env python3
"""Export a validated 50% mask checkpoint as a vLLM-loadable model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ramp_moe.checkpoints import apply_mask_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="4GB")
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


def save_model_assets(model_source: Path, output_dir: Path) -> list[str]:
    """Preserve the source processor, including tokenizer and multimodal assets."""

    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    config = AutoConfig.from_pretrained(model_source, local_files_only=True)
    architectures = tuple(getattr(config, "architectures", ()) or ())
    text_only = bool(architectures) and all(
        architecture.endswith("ForCausalLM") for architecture in architectures
    )
    asset_owner = (
        AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        if text_only
        else AutoProcessor.from_pretrained(model_source, local_files_only=True)
    )
    asset_owner.save_pretrained(output_dir)
    return sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in existing
    )


@torch.no_grad()
def apply_and_save_model(
    model: torch.nn.Module,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    expected_model: str,
    expected_revision: str,
    max_shard_size: str,
) -> dict[str, object]:
    """Apply a validated mask and save weights without overwriting a directory."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output_dir}")
    metadata = apply_mask_checkpoint(
        model,
        checkpoint_path,
        expected_model=expected_model,
        expected_revision=expected_revision,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    return metadata


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    model_source = args.model_source.resolve()
    checkpoint_path = args.mask_checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    metadata = apply_and_save_model(
        model,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        expected_model=args.model,
        expected_revision=args.revision,
        max_shard_size=args.max_shard_size,
    )
    processor_assets = save_model_assets(model_source, output_dir)

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    result = {
        "schema_version": 1,
        "experiment_type": "masked_model_export",
        "model": args.model,
        "model_revision": args.revision,
        "model_source": str(model_source),
        "mask_checkpoint": str(checkpoint_path),
        "mask_checkpoint_sha256": _sha256(checkpoint_path),
        "method": metadata["method"],
        "allocation": metadata["allocation"],
        "prune_ratio": metadata["prune_ratio"],
        "output_dir": str(output_dir),
        "weight_shards": shards,
        "weight_file_bytes": sum((output_dir / name).stat().st_size for name in shards),
        "processor_assets": processor_assets,
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "command": [sys.executable, *sys.argv],
        "status": "complete",
    }
    _json_atomic_write(output_dir / "ramp_moe_export.json", result)
    _json_atomic_write(args.result_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
