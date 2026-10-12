#!/usr/bin/env python3
"""Measure MATH generation length and repetition without changing evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ramp_moe.generation import (
    first_repeated_ngram_occurrence,
    repeated_ngram_fraction,
)

__all__ = ["first_repeated_ngram_occurrence", "repeated_ngram_fraction"]


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("prediction message content must be text or content parts")
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("prediction content part must be an object")
        part_type = part.get("type")
        if part_type == "reasoning":
            continue
        if part_type != "text" or not isinstance(part.get("text"), str):
            raise ValueError(f"unsupported prediction content part: {part_type!r}")
        text_parts.append(part["text"])
    return "\n".join(text_parts)


def prediction_diagnostic(
    record: dict[str, object],
    *,
    ngram_size: int = 8,
    near_cap_tokens: int = 2000,
    repetition_threshold: float = 0.25,
) -> dict[str, object]:
    """Extract one stable diagnostic row from an EvalScope prediction."""

    metadata = record.get("metadata")
    model_output = record.get("model_output")
    if not isinstance(metadata, dict) or not isinstance(model_output, dict):
        raise ValueError("prediction is missing metadata or model_output")
    question_id = metadata.get("question_id")
    choices = model_output.get("choices")
    usage = model_output.get("usage")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("prediction is missing question_id")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError("prediction must contain one choice and usage")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("prediction choice is missing message")
    content = _message_content_text(choice["message"].get("content"))
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, int):
        raise ValueError("prediction is missing text or integer output_tokens")
    repetition = repeated_ngram_fraction(content, n=ngram_size)
    return {
        "question_id": question_id,
        "output_tokens": output_tokens,
        "near_cap": output_tokens >= near_cap_tokens,
        "repeated_ngram_fraction": repetition,
        "high_repetition": repetition >= repetition_threshold,
    }


def summarize_diagnostics(
    diagnostics: Iterable[dict[str, object]],
) -> dict[str, float | int]:
    """Aggregate deterministic generation pathology metrics."""

    rows = list(diagnostics)
    if not rows:
        raise ValueError("at least one diagnostic is required")
    count = len(rows)
    return {
        "num_samples": count,
        "near_cap_rate": sum(bool(row["near_cap"]) for row in rows) / count,
        "high_repetition_rate": sum(
            bool(row["high_repetition"]) for row in rows
        )
        / count,
        "mean_output_tokens": sum(int(row["output_tokens"]) for row in rows)
        / count,
        "mean_repeated_ngram_fraction": sum(
            float(row["repeated_ngram_fraction"]) for row in rows
        )
        / count,
    }


def load_prediction_diagnostics(
    prediction_dir: Path,
    *,
    ngram_size: int,
    near_cap_tokens: int,
    repetition_threshold: float,
) -> list[dict[str, object]]:
    files = sorted(prediction_dir.rglob("math_500_*.jsonl"))
    if not files:
        raise ValueError("no MATH-500 prediction JSONL files found")
    rows: list[dict[str, object]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rows.append(
                    prediction_diagnostic(
                        json.loads(line),
                        ngram_size=ngram_size,
                        near_cap_tokens=near_cap_tokens,
                        repetition_threshold=repetition_threshold,
                    )
                )
    question_ids = [str(row["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("duplicate MATH question_id values found")
    return sorted(rows, key=lambda row: str(row["question_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--near-cap-tokens", type=int, default=2000)
    parser.add_argument("--repetition-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.expected_samples <= 0 or args.ngram_size <= 0 or args.near_cap_tokens <= 0:
        parser.error("sample, n-gram, and near-cap values must be positive")
    if not 0 <= args.repetition_threshold <= 1:
        parser.error("--repetition-threshold must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    rows = load_prediction_diagnostics(
        args.prediction_dir,
        ngram_size=args.ngram_size,
        near_cap_tokens=args.near_cap_tokens,
        repetition_threshold=args.repetition_threshold,
    )
    if len(rows) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} samples, found {len(rows)}"
        )
    result = {
        "schema_version": 1,
        "experiment_type": "math_generation_pathology",
        "result_label": "formal_diagnostic",
        "candidate": args.label,
        "protocol": {
            "expected_samples": args.expected_samples,
            "near_cap_threshold_tokens": args.near_cap_tokens,
            "high_repetition_threshold": args.repetition_threshold,
            "repetition_metric": (
                f"fraction of repeated whitespace {args.ngram_size}-grams"
            ),
        },
        "summary": summarize_diagnostics(rows),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
