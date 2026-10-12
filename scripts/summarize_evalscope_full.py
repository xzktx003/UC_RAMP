#!/usr/bin/env python3
"""Build an exact-count compact summary from one completed EvalScope full suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_FILES = {
    "piqa": ("piqa_default.jsonl",),
    "arc_easy": ("arc_ARC-Easy.jsonl",),
    "arc_challenge": ("arc_ARC-Challenge.jsonl",),
    "gsm8k": ("gsm8k_main.jsonl",),
    "hellaswag": ("hellaswag_default.jsonl",),
    "winogrande": ("winogrande_default.jsonl",),
    "race": ("race_high.jsonl", "race_middle.jsonl"),
    "gpqa_diamond": ("gpqa_diamond_default.jsonl",),
    "math500": tuple(f"math_500_Level {level}.jsonl" for level in range(1, 6)),
    "humaneval_pass_at_1": ("humaneval_openai_humaneval.jsonl",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--mask-result", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--ppl-result", type=Path, required=True)
    parser.add_argument("--math-pathology", type=Path, required=True)
    parser.add_argument("--terminal-exit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _review_file(eval_root: Path, served_name: str, filename: str) -> Path:
    matches = list(eval_root.glob(f"**/reviews/{served_name}/{filename}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one review file for {filename}, found {len(matches)}"
        )
    return matches[0]


def _exact_accuracy(paths: list[Path]) -> tuple[float, int, int]:
    correct = 0.0
    total = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                value = row["sample_score"]["score"]["value"]["acc"]
                correct += float(value)
                total += 1
    if total == 0:
        raise RuntimeError("review files contained no samples")
    return correct / total, int(correct), total


def main() -> None:
    args = parse_args()
    if args.terminal_exit.read_text(encoding="utf-8").strip() != "__EXPERIMENT_EXIT__=0":
        raise RuntimeError("full EvalScope terminal did not exit successfully")

    mask = _load_json(args.mask_result)
    ppl = _load_json(args.ppl_result)
    pathology = _load_json(args.math_pathology)
    if mask["status"] != "complete":
        raise RuntimeError("mask generation is incomplete")

    metrics: dict[str, float] = {
        "wikitext2_ppl_diagnostic": float(ppl["metrics"]["ppl"]),
    }
    sample_counts: dict[str, int] = {}
    correct_counts: dict[str, int] = {}
    for task, filenames in TASK_FILES.items():
        paths = [
            _review_file(args.eval_root, args.served_name, filename)
            for filename in filenames
        ]
        accuracy, correct, total = _exact_accuracy(paths)
        metrics[task] = accuracy
        sample_counts[task] = total
        correct_counts[task] = correct

    task_metric_names = tuple(TASK_FILES)
    metrics["accuracy_macro_average"] = sum(
        metrics[name] for name in task_metric_names
    ) / len(task_metric_names)
    pathology_summary = pathology["summary"]
    metrics["math500_high_repetition_rate"] = float(
        pathology_summary["high_repetition_rate"]
    )
    metrics["math500_near_cap_rate"] = float(pathology_summary["near_cap_rate"])

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_type": "review_v1_fair_ablation_evalscope_summary",
        "model": args.model_label,
        "method": args.method_label,
        "profile": mask["profile"],
        "calibration_aggregation": mask["calibration_aggregation"],
        "max_swaps": mask["max_swaps"],
        "granularity": "channel",
        "prune_ratio": mask["prune_ratio"],
        "retained_channels": mask["retained_channels"],
        "total_channels": mask["available_channels"],
        "mask_checkpoint_sha256": _sha256(args.mask_checkpoint),
        "protocol": {
            "evaluator": "EvalScope 1.4.1",
            "temperature": 0,
            "max_tokens": 2048,
            "seed": 1234,
            "mmlu_used": False,
            "offline": True,
            "calibration_windows": mask["nsamples"],
            "calibration_sequence_length": mask["sequence_length"],
            "layer_budget_schedule": mask["layer_budget_schedule"],
        },
        "metrics": metrics,
        "correct_counts": correct_counts,
        "sample_counts": sample_counts,
        "sources": {
            "mask_generation": str(args.mask_result),
            "ppl_result": str(args.ppl_result),
            "math_pathology": str(args.math_pathology),
            "eval_root": str(args.eval_root),
            "terminal_exit": str(args.terminal_exit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
