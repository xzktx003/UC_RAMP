import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mask_tool = load_tool("mask_artifact")
result_tool = load_tool("check_results")


def sample_checkpoint():
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    return {"schema_version": 1, "model": "test", "model_revision": "test",
            "prune_ratio": 0.25, "masks": [mask], "layer_budgets": [6]}


def test_mask_round_trip_excludes_source_paths():
    checkpoint = sample_checkpoint()
    checkpoint["fixed_stage1_checkpoint"] = "/private/example/reference.pt"
    artifact = mask_tool.encode_checkpoint(checkpoint)
    assert "fixed_stage1_checkpoint" not in artifact["metadata"]
    restored = mask_tool.decode_artifact(artifact)
    assert torch.equal(restored["masks"][0], checkpoint["masks"][0])


def test_mask_corruption_is_rejected():
    artifact = mask_tool.encode_checkpoint(sample_checkpoint())
    artifact["layers"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        mask_tool.decode_artifact(artifact)


def test_mask_budget_mismatch_is_rejected():
    artifact = mask_tool.encode_checkpoint(sample_checkpoint())
    artifact["metadata"]["prune_ratio"] = 0.5
    with pytest.raises(ValueError, match="budget"):
        mask_tool.decode_artifact(artifact)


def test_six_reference_result_counts_and_five_strict_masks():
    manifest = json.loads((ROOT / "reference/manifest.json").read_text())
    assert len(manifest["runs"]) == 6
    assert sum(run["mask_available"] for run in manifest["runs"].values()) == 5
    for run in manifest["runs"].values():
        summary = json.loads((ROOT / run["summary"]).read_text())
        assert result_tool.compare(summary, summary) == []
        if run["mask_available"]:
            artifact = json.loads((ROOT / run["mask_artifact"]).read_text())
            checkpoint = mask_tool.decode_artifact(artifact)
            assert sum(int(mask.sum()) for mask in checkpoint["masks"]) == summary["retained_channels"]
            assert sum(mask.numel() for mask in checkpoint["masks"]) == summary["total_channels"]
            cape = mask_tool.decode_artifact(json.loads((ROOT / run["stage1_mask_artifact"]).read_text()))
            assert all(torch.equal(a.sum(1), b.sum(1)) for a, b in zip(checkpoint["masks"], cape["masks"]))


def test_results_checker_detects_one_task_regression():
    reference = json.loads((ROOT / "reference/results/qwen3_prune25.json").read_text())
    actual = copy.deepcopy(reference)
    actual["correct_counts"]["math500"] -= 1
    actual["metrics"]["math500"] -= 1 / actual["sample_counts"]["math500"]
    actual["metrics"]["accuracy_macro_average"] -= 1 / (500 * 10)
    assert any("math500" in failure for failure in result_tool.compare(actual, reference))


def test_task_prompt_snapshots_pin_fewshots_and_shuffle():
    for name in ("qwen3_prune25", "gemma4_prune25", "gemma4_prune50", "qwen36_prune25", "qwen36_prune50"):
        root = ROOT / "reference/task_configs" / name
        general = json.loads((root / "general.json").read_text())["dataset_args"]
        knowledge = json.loads((root / "knowledge.json").read_text())["dataset_args"]
        assert general["gsm8k"]["few_shot_num"] == 4
        assert knowledge["race"]["few_shot_num"] == 3
        assert all(not task["shuffle"] and not task["shuffle_choices"] for task in (*general.values(), *knowledge.values()))
