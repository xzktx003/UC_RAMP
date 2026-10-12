#!/usr/bin/env python3
"""Check all ten task scores, counts and macro accuracy against a fixed run."""
import argparse
import json
import math
from pathlib import Path


TASKS = ("piqa", "arc_easy", "arc_challenge", "gsm8k", "hellaswag", "winogrande",
         "race", "gpqa_diamond", "math500", "humaneval_pass_at_1")


def compare(actual, reference, tolerance=0.0):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    failures = []
    for payload in (actual, reference):
        if payload.get("status") != "complete":
            raise ValueError("summary is not complete")
        for task in TASKS:
            n = payload["sample_counts"][task]
            k = payload["correct_counts"][task]
            score = payload["metrics"][task]
            if not isinstance(n, int) or n <= 0 or not 0 <= k <= n:
                raise ValueError(f"invalid sample counts: {task}")
            if not math.isfinite(score) or abs(score - k / n) > 1e-12:
                raise ValueError(f"score/count mismatch: {task}")
        macro = sum(payload["metrics"][t] for t in TASKS) / len(TASKS)
        if abs(macro - payload["metrics"]["accuracy_macro_average"]) > 1e-12:
            raise ValueError("incorrect ten-task macro average")
    for key in ("prune_ratio", "retained_channels", "total_channels"):
        if actual[key] != reference[key]:
            failures.append(f"{key}: {actual[key]} != {reference[key]}")
    for task in TASKS:
        if actual["sample_counts"][task] != reference["sample_counts"][task]:
            failures.append(f"{task}: sample count differs")
        delta = actual["metrics"][task] - reference["metrics"][task]
        if abs(delta) > tolerance + 1e-12:
            failures.append(f"{task}: difference {delta:+.8f} exceeds tolerance {tolerance}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0,
                        help="Absolute per-task accuracy tolerance as a fraction; default exact")
    args = parser.parse_args()
    failures = compare(json.loads(args.actual.read_text()), json.loads(args.reference.read_text()),
                       args.accuracy_tolerance)
    print(json.dumps({"status": "different" if failures else "matched", "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
