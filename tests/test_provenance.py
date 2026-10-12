import hashlib
import json

import pytest

from ramp_moe.provenance import verify_weight_manifest


def test_weight_manifest_verifies_size_and_sha256(tmp_path) -> None:
    payload = b"exact model bytes"
    (tmp_path / "shard.bin").write_bytes(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "shard.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        )
    )

    result = verify_weight_manifest(tmp_path, manifest_path)

    assert result["verified_files"] == 1
    assert result["verified_bytes"] == len(payload)


def test_weight_manifest_rejects_hash_mismatch(tmp_path) -> None:
    (tmp_path / "shard.bin").write_bytes(b"wrong")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {"name": "shard.bin", "size": 5, "sha256": "0" * 64}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="SHA256"):
        verify_weight_manifest(tmp_path, manifest_path)
