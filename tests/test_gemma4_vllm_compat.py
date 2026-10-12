from pathlib import Path


def test_gemma4_vllm_compat_is_opt_in_and_restores_legacy_fields() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compat"
        / "sitecustomize.py"
    ).read_text(encoding="utf-8")

    assert "UC_RAMP_GEMMA4_VLLM_COMPAT" in source
    assert "allow_global_per_layer_attribute_access = True" in source
    assert "global_head_dim" in source
    assert "num_global_key_value_heads" in source
