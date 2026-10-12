#!/usr/bin/env python3
"""Fingerprint local model assets without copying model weights."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root, hash_weights=False):
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "processor_config.json", "model.safetensors.index.json")
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
             for name in names if (root / name).is_file()}
    index = root / "model.safetensors.index.json"
    weights = {}
    if index.is_file():
        for name in sorted(set(json.loads(index.read_text())["weight_map"].values())):
            path = root / name
            weights[name] = {"bytes": path.stat().st_size}
            if hash_weights:
                weights[name]["sha256"] = sha256(path)
    return {"schema_version": 1, "scope": "current local assets; not a historical weight attestation",
            "files": files, "weight_shards": weights, "weight_hashes_included": hash_weights}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("choose exactly one of --output or --verify")
    payload = snapshot(args.model_source, args.hash_weights)
    if args.verify:
        reference = json.loads(args.verify.read_text())
        failures = [name for name, meta in reference["files"].items()
                    if payload["files"].get(name) != meta]
        for name, meta in reference["weight_shards"].items():
            current = payload["weight_shards"].get(name)
            if current is None or current["bytes"] != meta["bytes"]:
                failures.append(name)
            elif "sha256" in meta and sha256(args.model_source / name) != meta["sha256"]:
                failures.append(name)
        print(json.dumps({"status": "different" if failures else "matched", "files": failures}))
        raise SystemExit(1 if failures else 0)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
