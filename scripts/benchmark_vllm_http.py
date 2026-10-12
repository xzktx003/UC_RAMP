"""HTTP streaming benchmark for BF16 full vLLM versus compact Qwen3 vLLM.

This harness measures the OpenAI-compatible ``/v1/completions`` streaming
path, not the offline ``LLM.generate`` proxy used by
``benchmark_vllm_compact.py``.  Run it once per mode so full and compact
weights live in separate server processes:

```
python scripts/benchmark_vllm_http.py --mode full --model-source /path/model \
  --output full.json --port 18001
python scripts/benchmark_vllm_http.py --mode compact --model-source /path/model \
  --mask-checkpoint /path/masks.pt --output compact.json --port 18002
```

The compact server is launched through a worker-side custom architecture:
the child process registers ``Qwen3MoeCompactForCausalLM`` before invoking
``vllm serve`` and passes both ``--hf-overrides`` and
``--worker-extension-cls``.  If the server cannot be launched or readiness
does not complete, the script fails closed without producing benchmark claims.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib import error, request

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists():
    repo_src_text = str(REPO_SRC)
    if repo_src_text not in sys.path:
        sys.path.insert(0, repo_src_text)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in current_pythonpath.split(os.pathsep) if part]
    if repo_src_text not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_src_text, *pythonpath_parts])

with contextlib.redirect_stdout(sys.stderr):
    from ramp_moe.vllm_compact import (  # noqa: E402
        AUTO_REGISTER_ENV,
        COMPACT_QWEN3_ARCH,
        MASK_CHECKPOINT_ENV,
        load_masks_from_checkpoint,
        register_compact_qwen3_model,
    )

WORKLOADS: tuple[tuple[int, int], ...] = (
    (256, 1),
    (512, 1),
    (1024, 1),
    (1024, 2),
    (1024, 4),
)


@dataclass(frozen=True)
class ChoiceStreamMetrics:
    index: int
    ttft_ms: float | None
    finish_ms: float | None
    tpot_ms: float | None
    output_tokens: int
    token_source: str
    valid: bool
    invalid_reason: str | None
    chunk_count: int
    output_text: str


@dataclass(frozen=True)
class BatchStreamMetrics:
    request_ms: float
    choices: list[ChoiceStreamMetrics]
    usage: dict[str, Any] | None


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute median of an empty sequence")
    return float(statistics.median(values))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def build_server_child_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--server-child",
        "--mode",
        args.mode,
        "--model-source",
        str(args.model_source),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.mode == "compact":
        cmd.extend(["--mask-checkpoint", str(args.mask_checkpoint)])
    for extra in args.extra_vllm_arg:
        cmd.append(f"--extra-vllm-arg={extra}")
    return cmd


def build_vllm_serve_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "vllm",
        "serve",
        str(args.model_source),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.mode == "compact":
        argv.extend(
            [
                "--hf-overrides",
                json.dumps({"architectures": [COMPACT_QWEN3_ARCH]}),
                "--worker-extension-cls",
                "ramp_moe.vllm_compact.CompactModelRegistryWorkerExtension",
            ]
        )
    argv.extend(args.extra_vllm_arg)
    return argv


def configure_child_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if REPO_SRC.exists():
        parts = [str(REPO_SRC)]
        parts.extend(part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    if args.mode == "compact":
        masks = load_masks_from_checkpoint(args.mask_checkpoint)
        if not masks:
            raise ValueError("mask checkpoint did not contain any masks")
        env[MASK_CHECKPOINT_ENV] = str(args.mask_checkpoint.resolve())
        env[AUTO_REGISTER_ENV] = "1"
    return env


def run_server_child(args: argparse.Namespace) -> None:
    os.environ.update(configure_child_environment(args))
    if args.mode == "compact":
        register_compact_qwen3_model()
    sys.argv = build_vllm_serve_argv(args)
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


def start_server(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    env = configure_child_environment(args)
    cmd = build_server_child_command(args)
    return subprocess.Popen(cmd, env=env)


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def server_url(args: argparse.Namespace) -> str:
    if args.server_url:
        return args.server_url.rstrip("/")
    return f"http://{args.host}:{args.port}"


def wait_for_server_ready(
    base_url: str,
    *,
    timeout_s: float,
    process: subprocess.Popen[bytes] | None,
) -> None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{base_url}/v1/models"
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before readiness with code {process.returncode}"
            )
        try:
            with request.urlopen(models_url, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server was not ready after {timeout_s}s: {last_error}")


def parse_sse_line(raw_line: bytes) -> Any | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return "[DONE]"
    return json.loads(payload)


def extract_choice_text(choice: dict[str, Any]) -> str:
    if "text" in choice and choice["text"] is not None:
        return str(choice["text"])
    delta = choice.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta["content"])
    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    return ""


def extract_choice_token_ids(choice: dict[str, Any]) -> list[int] | None:
    token_ids = choice.get("token_ids")
    if token_ids is None:
        return None
    if not isinstance(token_ids, list):
        return None
    return [int(token_id) for token_id in token_ids]


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def metrics_from_sse_events(
    events: Sequence[tuple[float, Any]],
    *,
    start_time: float,
    tokenizer: Any,
    batch_size: int,
    expected_decode_tokens: int,
) -> BatchStreamMetrics:
    text_by_index: dict[int, list[str]] = {idx: [] for idx in range(batch_size)}
    token_ids_by_index: dict[int, list[int]] = {idx: [] for idx in range(batch_size)}
    saw_token_ids_by_index: dict[int, bool] = {idx: False for idx in range(batch_size)}
    first_by_index: dict[int, float] = {}
    last_by_index: dict[int, float] = {}
    chunks_by_index: dict[int, int] = {idx: 0 for idx in range(batch_size)}
    usage: dict[str, Any] | None = None
    done_time = events[-1][0] if events else start_time

    for event_time, payload in events:
        done_time = event_time
        if payload == "[DONE]":
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        for choice in payload.get("choices", []) or []:
            index = int(choice.get("index", 0))
            if index not in text_by_index:
                text_by_index[index] = []
                token_ids_by_index[index] = []
                saw_token_ids_by_index[index] = False
                chunks_by_index[index] = 0
            text = extract_choice_text(choice)
            token_ids = extract_choice_token_ids(choice)
            if token_ids is not None:
                saw_token_ids_by_index[index] = True
                token_ids_by_index[index].extend(token_ids)
            if not text and not token_ids:
                continue
            text_by_index[index].append(text)
            chunks_by_index[index] += 1
            first_by_index.setdefault(index, event_time)
            last_by_index[index] = event_time

    choices: list[ChoiceStreamMetrics] = []
    for index in range(batch_size):
        output_text = "".join(text_by_index.get(index, []))
        if saw_token_ids_by_index.get(index, False):
            output_tokens = len(token_ids_by_index.get(index, []))
            token_source = "sse_token_ids"
        else:
            output_tokens = count_tokens(tokenizer, output_text)
            token_source = "retokenized_estimate"
        first = first_by_index.get(index)
        last = last_by_index.get(index)
        invalid_reason = None
        if token_source != "sse_token_ids":
            invalid_reason = "missing_sse_token_ids"
        elif output_tokens != expected_decode_tokens:
            invalid_reason = (
                f"expected_{expected_decode_tokens}_output_tokens_got_{output_tokens}"
            )
        valid = invalid_reason is None
        if valid and first is not None:
            ttft_ms = (first - start_time) * 1000.0
        else:
            ttft_ms = None
        if valid and last is not None:
            finish_ms = (last - start_time) * 1000.0
        else:
            finish_ms = None
        if valid and first is not None and last is not None and output_tokens > 1:
            tpot_ms = (last - first) * 1000.0 / max(output_tokens - 1, 1)
        else:
            tpot_ms = None
        choices.append(
            ChoiceStreamMetrics(
                index=index,
                ttft_ms=ttft_ms,
                finish_ms=finish_ms,
                tpot_ms=tpot_ms,
                output_tokens=output_tokens,
                token_source=token_source,
                valid=valid,
                invalid_reason=invalid_reason,
                chunk_count=chunks_by_index.get(index, 0),
                output_text=output_text,
            )
        )

    return BatchStreamMetrics(
        request_ms=(done_time - start_time) * 1000.0,
        choices=choices,
        usage=usage,
    )


def stream_completion_request(
    *,
    base_url: str,
    served_model_name: str,
    prompt_token_id: int,
    prompt_tokens: int,
    batch_size: int,
    decode_tokens: int,
    timeout_s: float,
    tokenizer: Any,
) -> BatchStreamMetrics:
    payload = {
        "model": served_model_name,
        "prompt": [[prompt_token_id] * prompt_tokens for _ in range(batch_size)],
        "max_tokens": decode_tokens,
        "min_tokens": decode_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.perf_counter()
    events: list[tuple[float, Any]] = []
    with request.urlopen(req, timeout=timeout_s) as response:
        if response.status != 200:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"completion request failed: HTTP {response.status}: {body}")
        for raw_line in response:
            parsed = parse_sse_line(raw_line)
            if parsed is None:
                continue
            now = time.perf_counter()
            events.append((now, parsed))
            if parsed == "[DONE]":
                break
    return metrics_from_sse_events(
        events,
        start_time=start_time,
        tokenizer=tokenizer,
        batch_size=batch_size,
        expected_decode_tokens=decode_tokens,
    )


def load_tokenizer(model_source: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_source),
        trust_remote_code=False,
        local_files_only=True,
    )


def choice_to_json(choice: ChoiceStreamMetrics, *, include_text: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": choice.index,
        "ttft_ms": choice.ttft_ms,
        "finish_ms": choice.finish_ms,
        "tpot_ms": choice.tpot_ms,
        "output_tokens": choice.output_tokens,
        "token_source": choice.token_source,
        "valid": choice.valid,
        "invalid_reason": choice.invalid_reason,
        "chunk_count": choice.chunk_count,
    }
    if include_text:
        record["output_text"] = choice.output_text
    return record


def summarize_batch_trials(trials: Sequence[BatchStreamMetrics]) -> dict[str, Any]:
    invalid = [
        choice
        for trial in trials
        for choice in trial.choices
        if not choice.valid
    ]
    if invalid:
        raise ValueError(
            "cannot summarize invalid streaming trials: "
            + "; ".join(
                f"choice_{choice.index}:{choice.invalid_reason}" for choice in invalid
            )
        )
    ttfts = [
        choice.ttft_ms
        for trial in trials
        for choice in trial.choices
        if choice.ttft_ms is not None
    ]
    tpots = [
        choice.tpot_ms
        for trial in trials
        for choice in trial.choices
        if choice.tpot_ms is not None
    ]
    request_ms = [trial.request_ms for trial in trials]
    output_tokens = [
        choice.output_tokens for trial in trials for choice in trial.choices
    ]
    return {
        "median_request_ms": median(request_ms),
        "median_ttft_ms": median(ttfts) if ttfts else None,
        "p90_ttft_ms": percentile(ttfts, 0.90) if ttfts else None,
        "median_tpot_ms": median(tpots) if tpots else None,
        "median_output_tokens": median(output_tokens) if output_tokens else 0.0,
    }


def validate_batch_metrics(
    metrics: BatchStreamMetrics,
    *,
    prompt_tokens: int,
    batch_size: int,
    decode_tokens: int,
) -> None:
    if len(metrics.choices) != batch_size:
        raise RuntimeError(
            f"expected {batch_size} choices for prompt={prompt_tokens}, "
            f"got {len(metrics.choices)}"
        )
    failures = [
        f"choice={choice.index} reason={choice.invalid_reason} "
        f"tokens={choice.output_tokens}"
        for choice in metrics.choices
        if not choice.valid
    ]
    if failures:
        raise RuntimeError(
            f"invalid HTTP streaming output for prompt={prompt_tokens}, "
            f"batch={batch_size}, decode={decode_tokens}: " + "; ".join(failures)
        )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model_source)
    base_url = server_url(args)
    process: subprocess.Popen[bytes] | None = None

    try:
        if args.server_url is None:
            process = start_server(args)
            wait_for_server_ready(
                base_url,
                timeout_s=args.server_start_timeout,
                process=process,
            )
        else:
            wait_for_server_ready(
                base_url,
                timeout_s=args.server_start_timeout,
                process=None,
            )

        results: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "pair_id": args.pair_id,
            "model_source": str(args.model_source),
            "mask_checkpoint": str(args.mask_checkpoint) if args.mask_checkpoint else None,
            "server_url": base_url,
            "served_model_name": args.served_model_name,
            "timing_method": "true_http_sse_streaming_per_request_timestamps",
            "decode_tokens": args.decode_tokens,
            "prompt_token_id": args.prompt_token_id,
            "token_count_source": "sse_token_ids_required",
            "workloads": [],
            "note": (
                "No numbers in this JSON should be treated as final until produced by "
                "an actual GPU run with this script."
            ),
        }

        for prompt_tokens, batch_size in WORKLOADS:
            for _ in range(args.warmup_repeats):
                warmup = stream_completion_request(
                    base_url=base_url,
                    served_model_name=args.served_model_name,
                    prompt_token_id=args.prompt_token_id,
                    prompt_tokens=prompt_tokens,
                    batch_size=batch_size,
                    decode_tokens=args.decode_tokens,
                    timeout_s=args.request_timeout,
                    tokenizer=tokenizer,
                )
                validate_batch_metrics(
                    warmup,
                    prompt_tokens=prompt_tokens,
                    batch_size=batch_size,
                    decode_tokens=args.decode_tokens,
                )
            trials = []
            for _ in range(args.repeats):
                trial = stream_completion_request(
                    base_url=base_url,
                    served_model_name=args.served_model_name,
                    prompt_token_id=args.prompt_token_id,
                    prompt_tokens=prompt_tokens,
                    batch_size=batch_size,
                    decode_tokens=args.decode_tokens,
                    timeout_s=args.request_timeout,
                    tokenizer=tokenizer,
                )
                validate_batch_metrics(
                    trial,
                    prompt_tokens=prompt_tokens,
                    batch_size=batch_size,
                    decode_tokens=args.decode_tokens,
                )
                trials.append(trial)
            results["workloads"].append(
                {
                    "prompt_tokens": prompt_tokens,
                    "batch_size": batch_size,
                    "summary": summarize_batch_trials(trials),
                    "trials": [
                        {
                            "request_ms": trial.request_ms,
                            "usage": trial.usage,
                            "choices": [
                                choice_to_json(choice, include_text=args.include_output_text)
                                for choice in trial.choices
                            ],
                        }
                        for trial in trials
                    ],
                }
            )
        return results
    finally:
        stop_server(process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "compact"), required=True)
    parser.add_argument("--mask-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--served-model-name", default="benchmark-model")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--server-start-timeout", type=float, default=900.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prompt-token-id", type=int, default=0)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--include-output-text", action="store_true")
    parser.add_argument("--dry-run-command", action="store_true")
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="extra raw argument passed to vllm serve; repeat for each token",
    )
    args = parser.parse_args()
    if not args.model_source.exists():
        parser.error(f"model source does not exist: {args.model_source}")
    if args.mode == "compact" and args.mask_checkpoint is None:
        parser.error("compact mode requires --mask-checkpoint")
    if args.mask_checkpoint is not None and not args.mask_checkpoint.exists():
        parser.error(f"mask checkpoint does not exist: {args.mask_checkpoint}")
    if args.decode_tokens < 2:
        parser.error("--decode-tokens must be >= 2")
    if args.repeats < 1 or args.warmup_repeats < 0:
        parser.error("--repeats must be positive and --warmup-repeats non-negative")
    if args.max_model_len < max(prompt for prompt, _ in WORKLOADS) + args.decode_tokens:
        parser.error("--max-model-len must cover the largest prompt plus decode tokens")
    if args.prompt_token_id < 0:
        parser.error("--prompt-token-id must be non-negative")
    if not args.server_child and not args.dry_run_command:
        if args.output is None:
            parser.error("--output is required unless --server-child or --dry-run-command")
        if args.output.exists():
            parser.error("refusing to overwrite existing output")
    return args


def main() -> None:
    args = parse_args()
    if args.server_child:
        run_server_child(args)
        return
    if args.dry_run_command:
        print(
            json.dumps(
                {
                    "server_command": build_server_child_command(args),
                    "vllm_argv": build_vllm_serve_argv(args),
                    "env": {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        MASK_CHECKPOINT_ENV: (
                            str(args.mask_checkpoint.resolve())
                            if args.mode == "compact"
                            else None
                        ),
                        AUTO_REGISTER_ENV: "1" if args.mode == "compact" else None,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    results = run_benchmark(args)
    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
