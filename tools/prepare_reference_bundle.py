#!/usr/bin/env python3
"""Package audited result metadata and Boolean masks from a research workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mask_artifact import decode_artifact, encode_checkpoint


RUNS = {
    "qwen3_prune25": ("v11_ucramp_qwen3_exact75", "v11_cape_qwen3_exact75"),
    "qwen3_prune50": ("review_v1_fixed_stage2_beta020_qwen3_exact50", "review_v1_fair_review_stage1_envelope_qwen3_exact50"),
    "gemma4_prune25": ("v11_ucramp_gemma4_exact75", "v11_cape_gemma4_exact75"),
    "gemma4_prune50": ("v11_ucramp_gemma4_exact50", "v11_cape_gemma4_exact50"),
    "qwen36_prune25": ("v11_ucramp_qwen36_exact75", "v11_cape_qwen36_exact75"),
    "qwen36_prune50": ("v11_ucramp_qwen36_exact50", "v11_cape_qwen36_exact50"),
}


def clean(value):
    if isinstance(value, dict):
        result = {key: clean(item) for key, item in value.items() if key != "local_reconstruction"}
        if isinstance(value.get("local_reconstruction"), list):
            result["local_reconstruction_record_count"] = len(value["local_reconstruction"])
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<local>/" + Path(value).name
    return value


def write(path: Path, payload):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--trim-generated-telemetry", action="store_true",
                        help="Compact an existing generated metadata bundle; preserve method settings")
    args = parser.parse_args()
    source, destination = args.source, args.destination
    if args.trim_generated_telemetry:
        for path in (destination / "reference/pruning").glob("*.json"):
            payload = clean(json.loads(path.read_text()))
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    manifest = {"schema_version": 1, "runs": {}}
    for run_id, (label, cape_label) in RUNS.items():
        results = source / "experiments/results"
        summary_suffix = "_evalscope_full_summary.json" if run_id == "qwen3_prune50" else "_evalscope_summary.json"
        summary = json.loads((results / (label + summary_suffix)).read_text())
        maskgen = json.loads((results / (label + "_maskgen.json")).read_text())
        if summary["status"] != "complete" or maskgen["status"] != "complete":
            raise ValueError(f"incomplete run: {run_id}")
        summary["sources"] = {"mask_generation": f"reference/pruning/{run_id}.json",
                              "ppl_result": f"reference/ppl/{run_id}.json"}
        write(destination / f"reference/results/{run_id}.json", clean(summary))
        write(destination / f"reference/pruning/{run_id}.json", clean(maskgen))
        write(destination / f"reference/ppl/{run_id}.json",
              clean(json.loads((results / (label + "_wikitext2.json")).read_text())))
        cape_meta = json.loads((results / (cape_label + "_maskgen.json")).read_text())
        write(destination / f"reference/pruning/{run_id}_cape.json", clean(cape_meta))
        entry = {"architecture": run_id.split("_")[0], "prune_ratio": summary["prune_ratio"],
                 "model_name": maskgen["model"], "model_revision": maskgen["model_revision"],
                 "summary": f"reference/results/{run_id}.json", "maskgen": f"reference/pruning/{run_id}.json",
                 "stage1_maskgen": f"reference/pruning/{run_id}_cape.json",
                 "pruning_environment": maskgen["environment"],
                 "original_mask_checkpoint_sha256": summary["mask_checkpoint_sha256"],
                 "mask_available": False, "mask_artifact": None, "stage1_mask_artifact": None}
        for stage, checkpoint_label in (("strict", label), ("cape", cape_label)):
            checkpoint_path = source / "experiments/checkpoints" / (checkpoint_label + ".pt")
            if not checkpoint_path.is_file():
                continue
            raw_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if stage == "strict" and raw_sha != summary["mask_checkpoint_sha256"]:
                raise ValueError(f"historical checkpoint digest mismatch: {run_id}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            artifact = encode_checkpoint(checkpoint)
            decoded = decode_artifact(artifact)
            assert all(torch.equal(a, b) for a, b in zip(checkpoint["masks"], decoded["masks"]))
            filename = run_id if stage == "strict" else run_id + "_cape"
            rel = f"reference/masks/{filename}.json"
            write(destination / rel, artifact)
            entry["mask_artifact" if stage == "strict" else "stage1_mask_artifact"] = rel
            if stage == "strict":
                entry["mask_available"] = True
        entry["replay_status"] = "mask_bits_verified" if entry["mask_available"] else "original_mask_missing_rebuild_not_yet_verified"
        manifest["runs"][run_id] = entry
    write(destination / "reference/manifest.json", manifest)
    write(destination / "reference/calibration/competition_math_train_decontamination.json",
          json.loads((source / "experiments/results/competition_math_train_decontamination.json").read_text()))
    print(json.dumps({"runs": len(RUNS), "available_strict_masks": sum(e["mask_available"] for e in manifest["runs"].values())}))


if __name__ == "__main__":
    main()
