from __future__ import annotations

import json
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "reproduction" / "reproduce_uc_ramp.py"

spec = importlib.util.spec_from_file_location("reproduce_uc_ramp", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["reproduce_uc_ramp"] = runner
spec.loader.exec_module(runner)


def run_dry(*args: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--repo-root",
            str(ROOT),
            "--model-source",
            "/models/local",
            "--competition-math-source",
            "/datasets/competition_math.parquet",
            "--competition-math-exclusion-manifest",
            "/datasets/decontam.json",
            *args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(completed.stdout)


def test_dry_run_builds_strict_cape_then_frozen_ramp() -> None:
    payload = run_dry("--arch", "qwen3", "--pruning-rate", "0.50", "--phase", "mask")

    commands = payload["commands"]
    assert [item["name"] for item in commands] == [
        "stage1_cape_mask",
        "stage2_strict_ramp",
    ]
    stage1 = " ".join(commands[0]["argv"])
    stage2 = " ".join(commands[1]["argv"])
    assert "--profile review_stage1_envelope" in stage1
    assert "--profile review_stage1_ramp" in stage2
    expected_checkpoint = ROOT / "experiments" / "checkpoints" / "uc_ramp_qwen3_exact50_cape.pt"
    assert f"--fixed-stage1-checkpoint {expected_checkpoint}" in stage2
    assert payload["settings"]["interaction_strength"] == 0.2
    assert payload["settings"]["max_swaps_per_expert"] == 1


def test_qwen36_manifest_pins_language_only_evalscope_and_task_defaults() -> None:
    payload = run_dry(
        "--arch",
        "qwen36",
        "--pruning-rate",
        "0.25",
        "--general-dataset-dir",
        "/datasets/general",
        "--knowledge-dataset-dir",
        "/datasets/knowledge",
        "--phase",
        "eval",
        "--phase",
        "summarize",
    )

    evalscope = payload["settings"]["evalscope"]
    assert evalscope["qwen36_language_model_only"] is True
    assert evalscope["mmlu"] is False
    assert evalscope["temperature"] == 0
    assert evalscope["max_tokens"] == 2048
    assert evalscope["dataset_args"]["general"]["gsm8k"]["few_shot_num"] == 4
    assert evalscope["dataset_args"]["knowledge"]["race"]["few_shot_num"] == 3
    assert evalscope["dataset_args"]["math500"]["math_500"]["shuffle"] is False


def test_reference_mask_dry_run_skips_rebuild_commands() -> None:
    payload = run_dry(
        "--arch",
        "gemma4",
        "--pruning-rate",
        "0.50",
        "--phase",
        "mask",
        "--use-reference-mask",
        "--reference-manifest",
        "reference/manifest.json",
        "--reference-run",
        "gemma4_prune50",
    )

    assert payload["commands"] == []
    assert payload["reference_import"] == {
        "manifest": str(ROOT / "reference" / "manifest.json"),
        "run": "gemma4_prune50",
        "tool": "tools/mask_artifact.py unpack",
    }


def test_full_dry_run_orders_eval_before_postprocessing() -> None:
    payload = run_dry(
        "--arch",
        "qwen3",
        "--pruning-rate",
        "0.25",
        "--general-dataset-dir",
        "/datasets/general",
        "--knowledge-dataset-dir",
        "/datasets/knowledge",
    )

    names = [item["name"] for item in payload["commands"]]
    assert names == [
        "stage1_cape_mask",
        "stage2_strict_ramp",
        "wikitext2_ppl",
        "export_masked_model",
        "verify_export",
        "evalscope_full",
        "math500_generation_diagnostics",
        "summarize_evalscope",
    ]


def test_reference_paths_resolve_from_repo_root_not_manifest_parent() -> None:
    args = SimpleNamespace(
        repo_root=ROOT,
        reference_manifest=ROOT / "reference" / "manifest.json",
    )

    path = runner.resolve_reference_path(args, "reference/masks/gemma4_prune50.json")

    assert path == ROOT / "reference" / "masks" / "gemma4_prune50.json"
    assert "reference/reference" not in str(path)


def test_unavailable_reference_mask_fails_even_in_dry_run() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--repo-root",
            str(ROOT),
            "--arch",
            "qwen3",
            "--pruning-rate",
            "0.50",
            "--model-source",
            "/models/local",
            "--phase",
            "mask",
            "--use-reference-mask",
            "--reference-manifest",
            "reference/manifest.json",
            "--reference-run",
            "qwen3_prune50",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert "does not include a redistributable mask" in completed.stderr


def test_reference_mask_sets_revision_and_does_not_require_calibration_paths() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--repo-root",
            str(ROOT),
            "--arch",
            "gemma4",
            "--pruning-rate",
            "0.50",
            "--model-source",
            "/models/local",
            "--phase",
            "mask",
            "--use-reference-mask",
            "--reference-manifest",
            "reference/manifest.json",
            "--reference-run",
            "gemma4_prune50",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["model_name"] == "local/gemma-4-26B-A4B-it"
    assert payload["reference_import"]["run"] == "gemma4_prune50"


def test_reference_revision_mismatch_is_rejected() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--repo-root",
            str(ROOT),
            "--arch",
            "gemma4",
            "--pruning-rate",
            "0.50",
            "--model-source",
            "/models/local",
            "--model-revision",
            "wrong",
            "--phase",
            "mask",
            "--use-reference-mask",
            "--reference-manifest",
            "reference/manifest.json",
            "--reference-run",
            "gemma4_prune50",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert "does not match reference" in completed.stderr


def test_runner_source_has_no_private_absolute_defaults() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "/data01/home/" not in source
    assert "/data01/datasets" not in source
