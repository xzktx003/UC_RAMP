#!/usr/bin/env python3
"""Extract evaluator settings without copying user paths or generated answers."""
import argparse
import json
from pathlib import Path

import yaml

from prepare_reference_bundle import RUNS, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    for run_id, (label, _) in RUNS.items():
        suffix = "_evalscope_full_summary.json" if run_id == "qwen3_prune50" else "_evalscope_summary.json"
        summary = json.loads((args.source / "experiments/results" / (label + suffix)).read_text())
        root = args.source / summary["sources"]["eval_root"]
        for group in ("general", "knowledge", "math500", "humaneval"):
            files = list((root / group).glob("**/configs/task_config_*.yaml"))
            if not files:
                continue
            if len(files) != 1:
                raise ValueError(f"ambiguous task config: {run_id}/{group}")
            config = yaml.safe_load(files[0].read_text())
            settings = {key: config.get(key) for key in (
                "dataset_args", "datasets", "dataset_hub", "generation_config",
                "eval_batch_size", "seed", "chat_template", "eval_type", "limit",
            )}
            write(args.destination / f"reference/task_configs/{run_id}/{group}.json", settings)


if __name__ == "__main__":
    main()
