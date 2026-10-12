from types import SimpleNamespace

import torch
from torch import nn

from ramp_moe.checkpoints import apply_mask_checkpoint
from test_qwen3_stats import _module


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = _module()

        self.model = SimpleNamespace(
            layers=nn.ModuleList([_Layer(), _Layer()])
        )


class _GemmaStyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        class _PackedExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.ones(2, 6, 4))
                self.down_proj = nn.Parameter(torch.ones(2, 4, 3))

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = nn.Linear(4, 2, bias=False)
                self.experts = _PackedExperts()

        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList([_Layer()]))
        )


def test_mask_checkpoint_is_validated_and_applied(tmp_path) -> None:
    model = _Model()
    masks = [
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.tensor([[False, True, True], [True, False, False]]),
    ]
    path = tmp_path / "masks.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model-id",
            "model_revision": "revision",
            "method": "camera",
            "allocation": "global",
            "prune_ratio": 0.5,
            "masks": masks,
        },
        path,
    )

    metadata = apply_mask_checkpoint(
        model,
        path,
        expected_model="model-id",
        expected_revision="revision",
    )

    assert metadata["method"] == "camera"
    for layer, mask in zip(model.model.layers, masks):
        for expert_index, expert in enumerate(layer.mlp.experts):
            pruned = ~mask[expert_index]
            assert torch.count_nonzero(expert.up_proj.weight[pruned]) == 0


def test_mask_checkpoint_accepts_registered_25_percent_budget(tmp_path) -> None:
    model = _Model()
    masks = [torch.ones((2, 3), dtype=torch.bool)] * 2
    path = tmp_path / "prune25.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model-id",
            "model_revision": "revision",
            "method": "camera",
            "allocation": "global",
            "prune_ratio": 0.25,
            "masks": masks,
        },
        path,
    )

    metadata = apply_mask_checkpoint(
        model, path, expected_model="model-id", expected_revision="revision"
    )

    assert metadata["prune_ratio"] == 0.25


def test_mask_checkpoint_rejects_unregistered_budget(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model-id",
            "model_revision": "revision",
            "method": "camera",
            "allocation": "global",
            "prune_ratio": 0.4,
            "masks": [],
        },
        path,
    )

    try:
        apply_mask_checkpoint(
            _Model(), path, expected_model="model-id", expected_revision="revision"
        )
    except ValueError as error:
        assert "25% or 50%" in str(error)
    else:
        raise AssertionError("unregistered checkpoints must be rejected")


def test_mask_checkpoint_applies_optional_down_coefficients(tmp_path) -> None:
    model = _Model()
    masks = [torch.tensor([[True, False, True], [False, True, False]])] * 2
    coefficients = [torch.tensor([[2.0, 1.0, 0.5], [1.0, 1.5, 1.0]])] * 2
    originals = [
        [expert.down_proj.weight.detach().clone() for expert in layer.mlp.experts]
        for layer in model.model.layers
    ]
    path = tmp_path / "coefficient_masks.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "model-id",
            "model_revision": "revision",
            "method": "ramp_camera",
            "allocation": "global",
            "prune_ratio": 0.5,
            "masks": masks,
            "down_coefficients": coefficients,
        },
        path,
    )

    apply_mask_checkpoint(
        model, path, expected_model="model-id", expected_revision="revision"
    )

    for layer_index, (layer, mask, coefficient) in enumerate(
        zip(model.model.layers, masks, coefficients)
    ):
        for expert_index, expert in enumerate(layer.mlp.experts):
            expected = originals[layer_index][expert_index]
            expected = expected * coefficient[expert_index].unsqueeze(0)
            expected[:, ~mask[expert_index]] = 0
            torch.testing.assert_close(expert.down_proj.weight, expected)


def test_mask_checkpoint_supports_gemma_style_packed_experts(tmp_path) -> None:
    model = _GemmaStyleModel()
    mask = torch.tensor([[True, False, True], [False, True, False]])
    path = tmp_path / "gemma_masks.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": "gemma-model-id",
            "model_revision": "revision",
            "method": "universal_envelope",
            "allocation": "global",
            "prune_ratio": 0.5,
            "masks": [mask],
        },
        path,
    )

    apply_mask_checkpoint(
        model,
        path,
        expected_model="gemma-model-id",
        expected_revision="revision",
    )

    experts = model.model.language_model.layers[0].experts
    for expert_index in range(2):
        pruned = ~mask[expert_index]
        assert torch.count_nonzero(
            experts.gate_up_proj[expert_index, :3][pruned]
        ) == 0
        assert torch.count_nonzero(experts.down_proj[expert_index, :, pruned]) == 0
