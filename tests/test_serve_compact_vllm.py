import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_compact_vllm.py"
spec = importlib.util.spec_from_file_location("serve_compact_vllm", MODULE_PATH)
serve = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = serve
spec.loader.exec_module(serve)


def make_args(tmp_path, *extra):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mask_path = tmp_path / "mask.pt"
    torch.save({"masks": [torch.ones(2, 3, dtype=torch.bool)]}, mask_path)
    old_argv = sys.argv
    sys.argv = [
        "serve_compact_vllm.py",
        "--model-source",
        str(model_dir),
        "--mask-checkpoint",
        str(mask_path),
        *extra,
    ]
    try:
        return serve.parse_args()
    finally:
        sys.argv = old_argv


def test_build_llm_kwargs_enforces_eager_by_default(tmp_path):
    args = make_args(tmp_path)

    kwargs = serve.build_llm_kwargs(args)

    assert kwargs["enforce_eager"] is True
    assert kwargs["max_model_len"] == 4096
    assert kwargs["hf_overrides"] == {"architectures": [serve.COMPACT_QWEN3_ARCH]}
    assert (
        kwargs["worker_extension_cls"]
        == "ramp_moe.vllm_compact.CompactModelRegistryWorkerExtension"
    )


def test_build_llm_kwargs_can_disable_eager_for_debug_only(tmp_path):
    args = make_args(tmp_path, "--no-enforce-eager")

    kwargs = serve.build_llm_kwargs(args)

    assert kwargs["enforce_eager"] is False


def test_build_llm_kwargs_allows_max_model_len_override(tmp_path):
    args = make_args(tmp_path, "--max-model-len", "8192")

    kwargs = serve.build_llm_kwargs(args)

    assert kwargs["max_model_len"] == 8192


def test_parse_args_rejects_non_positive_max_model_len(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mask_path = tmp_path / "mask.pt"
    torch.save({"masks": [torch.ones(2, 3, dtype=torch.bool)]}, mask_path)
    old_argv = sys.argv
    sys.argv = [
        "serve_compact_vllm.py",
        "--model-source",
        str(model_dir),
        "--mask-checkpoint",
        str(mask_path),
        "--max-model-len",
        "0",
    ]
    try:
        with pytest.raises(SystemExit):
            serve.parse_args()
    finally:
        sys.argv = old_argv


def test_dry_run_registration_reports_enforce_eager(tmp_path, capsys):
    pytest.importorskip("vllm")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mask_path = tmp_path / "mask.pt"
    torch.save({"masks": [torch.ones(2, 3, dtype=torch.bool)]}, mask_path)
    old_argv = sys.argv
    sys.argv = [
        "serve_compact_vllm.py",
        "--model-source",
        str(model_dir),
        "--mask-checkpoint",
        str(mask_path),
        "--dry-run-registration",
    ]
    try:
        serve.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "dry_run_complete"
    assert payload["enforce_eager"] is True
    assert payload["max_model_len"] == 4096
