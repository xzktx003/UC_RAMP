import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vllm_http.py"
spec = importlib.util.spec_from_file_location("benchmark_vllm_http", MODULE_PATH)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [piece for piece in text.strip().split(" ") if piece]


def test_parse_sse_line_handles_data_json_done_and_comments():
    assert benchmark.parse_sse_line(b"\n") is None
    assert benchmark.parse_sse_line(b": keepalive\n") is None

    parsed = benchmark.parse_sse_line(
        b'data: {"choices":[{"index":0,"text":"hello"}]}\n'
    )
    assert parsed == {"choices": [{"index": 0, "text": "hello"}]}
    assert benchmark.parse_sse_line(b"data: [DONE]\n") == "[DONE]"


def test_metrics_from_fake_sse_stream_tracks_per_choice_ttft_tpot_and_usage():
    events = [
        (
            101.0,
            {
                "choices": [
                    {"index": 0, "text": "a ", "token_ids": [11]},
                    {"index": 1, "text": "c ", "token_ids": [21]},
                ]
            },
        ),
        (
            102.0,
            {
                "choices": [
                    {"index": 0, "text": "b ", "token_ids": [12]},
                    {"index": 1, "text": "d ", "token_ids": [22]},
                ]
            },
        ),
        (102.5, {"usage": {"completion_tokens": 4}, "choices": []}),
        (103.0, "[DONE]"),
    ]

    metrics = benchmark.metrics_from_sse_events(
        events,
        start_time=100.0,
        tokenizer=FakeTokenizer(),
        batch_size=2,
        expected_decode_tokens=2,
    )

    assert metrics.request_ms == 3000.0
    assert metrics.usage == {"completion_tokens": 4}
    assert [choice.output_tokens for choice in metrics.choices] == [2, 2]
    assert [choice.token_source for choice in metrics.choices] == [
        "sse_token_ids",
        "sse_token_ids",
    ]
    assert [choice.valid for choice in metrics.choices] == [True, True]
    assert [choice.chunk_count for choice in metrics.choices] == [2, 2]
    assert [choice.ttft_ms for choice in metrics.choices] == [1000.0, 1000.0]
    assert [choice.finish_ms for choice in metrics.choices] == [2000.0, 2000.0]
    assert [choice.tpot_ms for choice in metrics.choices] == [1000.0, 1000.0]


def test_missing_sse_token_ids_is_retained_as_estimate_but_invalid():
    events = [
        (101.0, {"choices": [{"index": 0, "text": "a "}]}),
        (102.0, {"choices": [{"index": 0, "text": "b "}]}),
        (103.0, "[DONE]"),
    ]

    metrics = benchmark.metrics_from_sse_events(
        events,
        start_time=100.0,
        tokenizer=FakeTokenizer(),
        batch_size=1,
        expected_decode_tokens=2,
    )

    choice = metrics.choices[0]
    assert choice.output_tokens == 2
    assert choice.token_source == "retokenized_estimate"
    assert choice.valid is False
    assert choice.invalid_reason == "missing_sse_token_ids"
    assert choice.ttft_ms is None
    assert choice.tpot_ms is None
    with pytest.raises(ValueError, match="cannot summarize invalid"):
        benchmark.summarize_batch_trials([metrics])
    with pytest.raises(RuntimeError, match="missing_sse_token_ids"):
        benchmark.validate_batch_metrics(
            metrics,
            prompt_tokens=256,
            batch_size=1,
            decode_tokens=2,
        )


def test_short_sse_token_ids_invalidates_choice():
    events = [
        (101.0, {"choices": [{"index": 0, "text": "a ", "token_ids": [11]}]}),
        (102.0, "[DONE]"),
    ]

    metrics = benchmark.metrics_from_sse_events(
        events,
        start_time=100.0,
        tokenizer=FakeTokenizer(),
        batch_size=1,
        expected_decode_tokens=2,
    )

    choice = metrics.choices[0]
    assert choice.output_tokens == 1
    assert choice.valid is False
    assert choice.invalid_reason == "expected_2_output_tokens_got_1"
    with pytest.raises(RuntimeError, match="expected_2_output_tokens_got_1"):
        benchmark.validate_batch_metrics(
            metrics,
            prompt_tokens=256,
            batch_size=1,
            decode_tokens=2,
        )


def test_build_compact_server_command_and_vllm_argv_are_worker_side(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mask_path = tmp_path / "mask.pt"
    mask_path.write_bytes(b"placeholder")
    args = type(
        "Args",
        (),
        {
            "mode": "compact",
            "model_source": model_dir,
            "mask_checkpoint": mask_path,
            "host": "127.0.0.1",
            "port": 19000,
            "served_model_name": "bench",
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "extra_vllm_arg": ["--enforce-eager"],
        },
    )()

    child_cmd = benchmark.build_server_child_command(args)
    vllm_argv = benchmark.build_vllm_serve_argv(args)

    assert "--server-child" in child_cmd
    assert "--mask-checkpoint" in child_cmd
    assert "--extra-vllm-arg=--enforce-eager" in child_cmd
    assert "--extra-vllm-arg" not in child_cmd
    assert "--hf-overrides" in vllm_argv
    overrides = json.loads(vllm_argv[vllm_argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == [benchmark.COMPACT_QWEN3_ARCH]
    assert "--worker-extension-cls" in vllm_argv
    assert "ramp_moe.vllm_compact.CompactModelRegistryWorkerExtension" in vllm_argv
    assert "--enforce-eager" in vllm_argv


def test_extra_vllm_arg_child_command_roundtrips_dash_prefixed_value(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mask_path = tmp_path / "mask.pt"
    mask_path.write_bytes(b"placeholder")
    args = type(
        "Args",
        (),
        {
            "mode": "compact",
            "model_source": model_dir,
            "mask_checkpoint": mask_path,
            "host": "127.0.0.1",
            "port": 19000,
            "served_model_name": "bench",
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "extra_vllm_arg": ["--enforce-eager"],
        },
    )()
    child_cmd = benchmark.build_server_child_command(args)
    old_argv = sys.argv
    sys.argv = [child_cmd[1], *child_cmd[2:]]
    try:
        parsed = benchmark.parse_args()
    finally:
        sys.argv = old_argv

    assert parsed.server_child is True
    assert parsed.extra_vllm_arg == ["--enforce-eager"]


def test_installed_vllm_0271_completion_api_supports_required_streaming_fields():
    source_root = os.environ.get("UC_RAMP_VLLM_SOURCE")
    if source_root is None:
        pytest.skip("set UC_RAMP_VLLM_SOURCE to check a patched vLLM source tree")
    root = Path(source_root)
    protocol = root / "entrypoints/openai/completion/protocol.py"
    serving = root / "entrypoints/openai/completion/serving.py"
    if not protocol.exists() or not serving.exists():
        pytest.skip("installed gptqmodel vLLM source is not available")

    protocol_text = protocol.read_text(encoding="utf-8")
    serving_text = serving.read_text(encoding="utf-8")

    assert "list[list[Annotated[int" in protocol_text
    assert "stream_options: StreamOptions" in protocol_text
    assert "ignore_eos: bool" in protocol_text
    assert "min_tokens: int" in protocol_text
    assert "return_token_ids: bool" in protocol_text
    assert "if request.return_token_ids" in serving_text
    assert "token_ids=(" in serving_text
    assert "as_list(output.token_ids)" in serving_text


def test_summarize_batch_trials_uses_choice_level_values():
    metrics = benchmark.BatchStreamMetrics(
        request_ms=10.0,
        usage=None,
        choices=[
            benchmark.ChoiceStreamMetrics(
                index=0,
                ttft_ms=2.0,
                finish_ms=8.0,
                tpot_ms=3.0,
                output_tokens=3,
                token_source="sse_token_ids",
                valid=True,
                invalid_reason=None,
                chunk_count=2,
                output_text="a b c",
            )
        ],
    )

    summary = benchmark.summarize_batch_trials([metrics, metrics])

    assert summary["median_request_ms"] == 10.0
    assert summary["median_ttft_ms"] == 2.0
    assert summary["median_tpot_ms"] == 3.0
    assert summary["median_output_tokens"] == 3.0
