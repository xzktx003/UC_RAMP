"""Byte-level model-weight provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight_manifest(
    model_directory: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    """Verify every declared weight shard by exact size and SHA256."""

    model_directory = Path(model_directory)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("weight manifest must contain a non-empty files list")

    verified_bytes = 0
    verified: list[dict[str, object]] = []
    for entry in files:
        name = entry.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("manifest file names must be plain basenames")
        path = model_directory / name
        if not path.is_file():
            raise ValueError(f"missing model shard: {name}")
        actual_size = path.stat().st_size
        expected_size = entry.get("size")
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {name}: expected {expected_size}, got {actual_size}"
            )
        actual_sha256 = _sha256(path)
        expected_sha256 = entry.get("sha256")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {name}")
        verified_bytes += actual_size
        verified.append(
            {"name": name, "size": actual_size, "sha256": actual_sha256}
        )

    return {
        "model_directory": str(model_directory.resolve()),
        "manifest": str(manifest_path.resolve()),
        "verified_files": len(verified),
        "verified_bytes": verified_bytes,
        "files": verified,
    }
