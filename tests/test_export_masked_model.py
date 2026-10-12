from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from scripts.export_masked_model import apply_and_save_model, save_model_assets


def test_save_model_assets_preserves_multimodal_processor(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    calls: list[tuple[Path, bool]] = []

    class FakeProcessor:
        def save_pretrained(self, destination: Path) -> None:
            (destination / "processor_config.json").write_text(
                '{"processor_class":"FakeProcessor"}\n', encoding="utf-8"
            )

    def fake_from_pretrained(path: Path, *, local_files_only: bool):
        calls.append((Path(path), local_files_only))
        return FakeProcessor()

    monkeypatch.setattr(
        "scripts.export_masked_model.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: type(
            "Config", (), {"architectures": ["Gemma4ForConditionalGeneration"]}
        )(),
    )
    monkeypatch.setattr(
        "scripts.export_masked_model.AutoProcessor.from_pretrained",
        fake_from_pretrained,
    )

    saved = save_model_assets(source, output)

    assert calls == [(source, True)]
    assert saved == ["processor_config.json"]
    assert (output / "processor_config.json").is_file()


def test_save_model_assets_uses_tokenizer_for_text_causal_lm(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tokenizer_calls: list[tuple[Path, bool]] = []

    class FakeTokenizer:
        def save_pretrained(self, destination: Path) -> None:
            (destination / "tokenizer_config.json").write_text(
                '{"tokenizer_class":"FakeTokenizer"}\n', encoding="utf-8"
            )

    monkeypatch.setattr(
        "scripts.export_masked_model.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: type(
            "Config", (), {"architectures": ["Qwen3_5MoeForCausalLM"]}
        )(),
    )
    monkeypatch.setattr(
        "scripts.export_masked_model.AutoTokenizer.from_pretrained",
        lambda path, *, local_files_only: (
            tokenizer_calls.append((Path(path), local_files_only)) or FakeTokenizer()
        ),
    )
    monkeypatch.setattr(
        "scripts.export_masked_model.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("text-only export must not load a multimodal processor")
        ),
    )

    saved = save_model_assets(source, output)

    assert tokenizer_calls == [(source, True)]
    assert saved == ["tokenizer_config.json"]


def test_exported_model_reloads_with_joint_zero_mask(tmp_path: Path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )
    checkpoint = {
        "schema_version": 1,
        "model": "test/qwen3-moe",
        "model_revision": "test-revision",
        "method": "test",
        "allocation": "uniform",
        "prune_ratio": 0.5,
        "masks": [mask],
    }
    checkpoint_path = tmp_path / "mask.pt"
    torch.save(checkpoint, checkpoint_path)
    output_dir = tmp_path / "exported"

    apply_and_save_model(
        model,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        expected_model="test/qwen3-moe",
        expected_revision="test-revision",
        max_shard_size="1GB",
    )

    reloaded = Qwen3MoeForCausalLM.from_pretrained(
        output_dir, local_files_only=True
    )
    experts = reloaded.model.layers[0].mlp.experts
    if hasattr(experts, "gate_up_proj"):
        channel_count = mask.shape[1]
        for expert_index in range(mask.shape[0]):
            pruned = ~mask[expert_index]
            assert (
                torch.count_nonzero(
                    experts.gate_up_proj[expert_index, :channel_count][pruned]
                )
                == 0
            )
            assert (
                torch.count_nonzero(
                    experts.gate_up_proj[expert_index, channel_count:][pruned]
                )
                == 0
            )
            assert (
                torch.count_nonzero(experts.down_proj[expert_index, :, pruned])
                == 0
            )
    else:
        for expert_index, expert in enumerate(experts):
            pruned = ~mask[expert_index]
            assert torch.count_nonzero(expert.gate_proj.weight[pruned]) == 0
            assert torch.count_nonzero(expert.up_proj.weight[pruned]) == 0
            assert torch.count_nonzero(expert.down_proj.weight[:, pruned]) == 0
