#!/usr/bin/env python3
"""Run the pinned, non-vLLM WikiText-2 perplexity protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ramp_moe.evaluation import (
    WikiTextPPLConfig,
    dataset_download_config,
    evaluate_causal_lm_ppl,
)
from ramp_moe.checkpoints import apply_mask_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--model-source")
    parser.add_argument(
        "--tokenizer-source",
        help="Optional local tokenizer source for a physically pruned model export.",
    )
    parser.add_argument(
        "--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path)
    parser.add_argument("--method-label")
    parser.add_argument("--allocation-label")
    parser.add_argument("--prune-ratio", type=float)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--manual-cuda-load",
        action="store_true",
        help="Load on CPU and move to CUDA without Accelerate device_map support.",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "grouped_mm", "batched_mm"),
        help=(
            "Optional Transformers MoE expert backend. Use eager when the installed "
            "grouped-mm kernel does not support the visible GPU architecture."
        ),
    )
    return parser.parse_args()


def result_pruning_metadata(
    mask_metadata: dict[str, object] | None,
    *,
    method_label: str | None,
    allocation_label: str | None,
    prune_ratio: float | None,
) -> tuple[str, str | None, float | None]:
    """Resolve truthful metadata for masked and directly exported models."""

    if mask_metadata is not None:
        if any(value is not None for value in (method_label, allocation_label, prune_ratio)):
            raise ValueError("direct-model pruning labels cannot override mask metadata")
        return (
            str(mask_metadata["method"]),
            str(mask_metadata["allocation"]),
            float(mask_metadata["prune_ratio"]),
        )
    if prune_ratio is not None and not 0.0 <= prune_ratio <= 1.0:
        raise ValueError("prune_ratio must be in [0, 1]")
    return method_label or "dense", allocation_label, prune_ratio


def resolve_tokenizer_source(
    *,
    model_source: str | None,
    tokenizer_source: str | None,
    model_name: str,
    revision: str = "revision",
) -> tuple[str, str | None]:
    """Resolve an independently pinned tokenizer without changing model weights."""

    if tokenizer_source is not None:
        return tokenizer_source, None
    if model_source is not None:
        return model_source, None
    return model_name, revision


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "datasets", "evalscope", "vllm")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def model_loading_kwargs(
    *,
    source_revision: str | None,
    local_files_only: bool,
    experts_implementation: str | None,
) -> dict[str, object]:
    """Build model loading kwargs without changing the default MoE backend."""

    kwargs: dict[str, object] = {
        "revision": source_revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if experts_implementation is not None:
        kwargs["experts_implementation"] = experts_implementation
    return kwargs


def main() -> None:
    args = parse_args()
    model_source = args.model_source or args.model
    source_revision = None if args.model_source else args.revision
    tokenizer_source, tokenizer_revision = resolve_tokenizer_source(
        model_source=args.model_source,
        tokenizer_source=args.tokenizer_source,
        model_name=args.model,
        revision=args.revision,
    )
    config = WikiTextPPLConfig(
        seq_len=2048, smoke=args.smoke, max_windows=args.max_windows
    )
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        download_config=dataset_download_config(args.local_files_only),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=tokenizer_revision,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    input_ids = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids

    if not torch.cuda.is_available():
        raise RuntimeError("WikiText-2 model evaluation requires a visible CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    model_kwargs = model_loading_kwargs(
        source_revision=source_revision,
        local_files_only=args.local_files_only,
        experts_implementation=args.experts_implementation,
    )
    if args.manual_cuda_load:
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map={"": "cuda:0"},
            **model_kwargs,
        )
    mask_metadata = None
    if args.mask_checkpoint is not None:
        mask_metadata = apply_mask_checkpoint(
            model,
            args.mask_checkpoint,
            expected_model=args.model,
            expected_revision=args.revision,
        )
    metrics = evaluate_causal_lm_ppl(
        model, input_ids, config, input_device="cuda:0"
    )
    method, allocation, prune_ratio = result_pruning_metadata(
        mask_metadata,
        method_label=args.method_label,
        allocation_label=args.allocation_label,
        prune_ratio=args.prune_ratio,
    )

    result = {
        "schema_version": 1,
        "experiment_type": "wikitext2_ppl",
        "method": method,
        "allocation": allocation,
        "prune_ratio": prune_ratio,
        "mask_checkpoint": (
            None if args.mask_checkpoint is None else str(args.mask_checkpoint.resolve())
        ),
        "model": args.model,
        "model_revision": args.revision,
        "model_source": str(Path(model_source).resolve()) if args.model_source else model_source,
        "tokenizer_source": (
            str(Path(tokenizer_source).resolve())
            if args.tokenizer_source or args.model_source
            else tokenizer_source
        ),
        "dataset": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "cuda_visible_devices": __import__("os").environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
            "manual_cuda_load": bool(args.manual_cuda_load),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
