"""Run a local vLLM Qwen3 MoE instance with compact routed experts.

This script is intentionally import-safe when vLLM is absent.  It does not
download models; pass a local model path and a local mask checkpoint.  It
registers a custom Qwen3 architecture and asks vLLM workers to construct that
class directly, so compact experts are installed in the worker after weights
load instead of monkeypatching the main-process ``LLM`` object.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists():
    repo_src_text = str(REPO_SRC)
    if repo_src_text not in sys.path:
        sys.path.insert(0, repo_src_text)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in current_pythonpath.split(os.pathsep) if part]
    if repo_src_text not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_src_text, *pythonpath_parts])

from ramp_moe.vllm_compact import (
    AUTO_REGISTER_ENV,
    COMPACT_QWEN3_ARCH,
    MASK_CHECKPOINT_ENV,
    load_masks_from_checkpoint,
    register_compact_qwen3_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain why structured MoE pruning is hard.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help=(
            "vLLM maximum sequence length. Defaults to 4096 to avoid allocating "
            "the full model-advertised long-context KV cache during compact "
            "smoke tests."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run-registration", action="store_true")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run vLLM in eager mode. This is enabled by default because the "
            "compact ragged expert path uses data-dependent index validation "
            "that torch.compile cannot safely trace in vLLM 0.27.x."
        ),
    )
    args = parser.parse_args()
    if not args.model_source.exists():
        parser.error(f"model source does not exist: {args.model_source}")
    if not args.mask_checkpoint.exists():
        parser.error(f"mask checkpoint does not exist: {args.mask_checkpoint}")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    return args


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": str(args.model_source),
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": False,
        "download_dir": None,
        "max_model_len": args.max_model_len,
        "hf_overrides": {"architectures": [COMPACT_QWEN3_ARCH]},
        "worker_extension_cls": (
            "ramp_moe.vllm_compact.CompactModelRegistryWorkerExtension"
        ),
        "enforce_eager": bool(args.enforce_eager),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ[MASK_CHECKPOINT_ENV] = str(args.mask_checkpoint.resolve())
    os.environ[AUTO_REGISTER_ENV] = "1"
    masks = load_masks_from_checkpoint(args.mask_checkpoint)
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vLLM is not installed in this environment. Install/activate vLLM "
            "0.27.x, then rerun this script."
        ) from exc

    register_compact_qwen3_model()
    if args.dry_run_registration:
        print(
            json.dumps(
                {
                    "status": "dry_run_complete",
                    "backend": "vllm_worker_side_compact_registration",
                    "architecture": COMPACT_QWEN3_ARCH,
                    "model_source": str(args.model_source),
                    "mask_checkpoint": str(args.mask_checkpoint),
                    "mask_count": len(masks),
                    "first_mask_shape": list(masks[0].shape) if masks else None,
                    "enforce_eager": bool(args.enforce_eager),
                    "max_model_len": args.max_model_len,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    llm = LLM(**build_llm_kwargs(args))
    sampling = SamplingParams(max_tokens=args.max_tokens)
    outputs = llm.generate([args.prompt], sampling)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    result = {
        "status": "complete",
        "backend": "vllm_worker_side_compact_registration",
        "architecture": COMPACT_QWEN3_ARCH,
        "model_source": str(args.model_source),
        "mask_checkpoint": str(args.mask_checkpoint),
        "mask_count": len(masks),
        "enforce_eager": bool(args.enforce_eager),
        "max_model_len": args.max_model_len,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "text": text,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
