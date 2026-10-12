import pytest

from scripts.eval_wikitext import (
    model_loading_kwargs,
    package_versions,
    resolve_tokenizer_source,
    result_pruning_metadata,
)


def test_direct_external_model_can_record_pruning_metadata() -> None:
    assert result_pruning_metadata(
        None,
        method_label="reap_whole_expert",
        allocation_label="uniform_64_experts_per_layer",
        prune_ratio=0.5,
    ) == ("reap_whole_expert", "uniform_64_experts_per_layer", 0.5)


def test_mask_metadata_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="cannot override"):
        result_pruning_metadata(
            {"method": "camera", "allocation": "global", "prune_ratio": 0.5},
            method_label="other",
            allocation_label=None,
            prune_ratio=None,
        )


def test_direct_pruned_model_can_reuse_the_original_tokenizer() -> None:
    assert resolve_tokenizer_source(
        model_source="/models/pruned",
        tokenizer_source="/models/original",
        model_name="org/model",
    ) == ("/models/original", None)
    assert resolve_tokenizer_source(
        model_source=None,
        tokenizer_source=None,
        model_name="org/model",
    ) == ("org/model", "revision")


def test_optional_runtime_packages_are_reported_as_not_installed(monkeypatch) -> None:
    def fake_version(name: str) -> str:
        if name == "evalscope":
            raise __import__("importlib").metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr("scripts.eval_wikitext.importlib.metadata.version", fake_version)
    versions = package_versions()
    assert versions["evalscope"] == "not-installed"
    assert versions["torch"] == "1.0"


def test_model_loading_kwargs_can_force_eager_experts_backend() -> None:
    kwargs = model_loading_kwargs(
        source_revision=None,
        local_files_only=True,
        experts_implementation="eager",
    )

    assert kwargs["experts_implementation"] == "eager"
    assert kwargs["local_files_only"] is True


def test_model_loading_kwargs_preserves_default_backend_when_unspecified() -> None:
    kwargs = model_loading_kwargs(
        source_revision="revision",
        local_files_only=False,
        experts_implementation=None,
    )

    assert "experts_implementation" not in kwargs
