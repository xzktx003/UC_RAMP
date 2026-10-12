import pytest
import torch
from torch import nn
import weakref

from ramp_moe.vllm_compact import (
    COMPACT_QWEN3_ARCH,
    CompactVllmQwen3Experts,
    install_compact_qwen3_moe_,
    load_masks_from_checkpoint,
    pack_vllm_routed_experts,
    register_compact_qwen3_model,
)


class FakeGate(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hidden_states: torch.Tensor):
        return self.logits[: hidden_states.shape[0]].to(hidden_states.device), None


class FakeRoutedExperts(nn.Module):
    def __init__(self, w13: torch.Tensor, w2: torch.Tensor, top_k: int = 2):
        super().__init__()
        self.w13_weight = nn.Parameter(w13.clone(), requires_grad=False)
        self.w2_weight = nn.Parameter(w2.clone(), requires_grad=False)
        self.renormalize = True
        self.moe_config = type("FakeMoeConfig", (), {"experts_per_token": top_k})()


class AddOneSharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.ones_like(hidden_states)


def dense_masked_reference(
    x: torch.Tensor,
    logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, k=top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)
    result = torch.zeros_like(x)
    channels = mask.shape[1]
    for token in range(x.shape[0]):
        for route in range(top_k):
            expert = int(ids[token, route])
            selected = mask[expert].nonzero().flatten()
            gate = w13[expert, :channels].index_select(0, selected)
            up = w13[expert, channels:].index_select(0, selected)
            down = w2[expert].index_select(1, selected)
            activation = torch.nn.functional.silu(x[token] @ gate.T) * (x[token] @ up.T)
            result[token] += weights[token, route] * (activation @ down.T)
    return result


def test_pack_vllm_routed_experts_matches_dense_masked_reference():
    torch.manual_seed(13)
    experts, channels, hidden, tokens, top_k = 3, 5, 7, 4, 2
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    mask = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, True, False, False],
            [True, True, False, False, True],
        ]
    )
    logits = torch.randn(tokens, experts)
    x = torch.randn(tokens, hidden)
    compact = pack_vllm_routed_experts(FakeRoutedExperts(w13, w2, top_k), mask)
    module = CompactVllmQwen3Experts(
        compact,
        gate=FakeGate(logits),
        top_k=top_k,
        renormalize=True,
    )

    actual = module(hidden_states=x, router_logits=x)
    expected = dense_masked_reference(x, logits, w13, w2, mask, top_k)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert module.gate.shape[0] == int(mask.sum())
    assert module.up.shape[0] == int(mask.sum())
    assert module.down.shape[0] == int(mask.sum())


def test_pack_vllm_routed_experts_does_not_use_cat(monkeypatch):
    torch.manual_seed(14)
    experts, channels, hidden = 2, 4, 5
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])

    def fail_cat(*args, **kwargs):
        raise AssertionError("pack should preallocate instead of torch.cat")

    monkeypatch.setattr(torch, "cat", fail_cat)

    compact = pack_vllm_routed_experts(FakeRoutedExperts(w13, w2), mask)

    assert compact.gate.shape == (int(mask.sum()), hidden)
    assert compact.up.shape == (int(mask.sum()), hidden)
    assert compact.down.shape == (int(mask.sum()), hidden)


def test_pack_vllm_routed_experts_can_write_cpu_output_buffers():
    torch.manual_seed(15)
    experts, channels, hidden = 2, 4, 5
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])

    compact = pack_vllm_routed_experts(
        FakeRoutedExperts(w13, w2),
        mask,
        output_device=torch.device("cpu"),
    )

    assert compact.gate.device.type == "cpu"
    assert compact.up.device.type == "cpu"
    assert compact.down.device.type == "cpu"
    assert compact.offsets.device.type == "cpu"


def test_install_compact_qwen3_moe_replaces_fake_vllm_sparse_block():
    torch.manual_seed(17)
    experts, channels, hidden, tokens, top_k = 2, 4, 6, 3, 1
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    logits = torch.randn(tokens, experts)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate(logits)
            self.experts = FakeRoutedExperts(w13, w2, top_k=top_k)
            self.n_routed_experts = experts

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeVllmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer()])
            self.moe_layers = [self.model.layers[0].mlp.experts]

        def update_physical_experts_metadata(
            self,
            num_physical_experts: int,
            num_local_physical_experts: int,
        ) -> None:
            raise AssertionError("old metadata updater should be replaced")

    model = FakeVllmModel()
    old_runner = model.moe_layers[0]
    report = install_compact_qwen3_moe_(model, [mask])
    replacement = model.model.layers[0].mlp.experts

    assert report.layer_count == 1
    assert report.retained_channels == int(mask.sum())
    assert isinstance(replacement, CompactVllmQwen3Experts)
    assert model.moe_layers == [replacement]
    assert model.moe_layers[0] is not old_runner
    assert replacement.is_internal_router is True
    assert replacement.shared_experts is None
    with pytest.raises(NotImplementedError, match="physical expert metadata"):
        model.update_physical_experts_metadata(2, 2)
    with pytest.raises(NotImplementedError, match="expert-map updates"):
        replacement.update_expert_map()

    x = torch.randn(tokens, hidden)
    actual = replacement(hidden_states=x)
    expected = dense_masked_reference(x, logits, w13, w2, mask, top_k)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_install_compact_qwen3_moe_drops_stale_moe_layer_refs_incrementally(monkeypatch):
    torch.manual_seed(18)
    experts, channels, hidden, top_k = 2, 4, 6, 1
    logits = torch.randn(2, experts)
    masks = [
        torch.tensor([[True, False, True, False], [False, True, False, True]]),
        torch.tensor([[True, True, False, False], [False, False, True, True]]),
    ]

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate(logits)
            self.experts = FakeRoutedExperts(
                torch.randn(experts, 2 * channels, hidden),
                torch.randn(experts, hidden, channels),
                top_k=top_k,
            )
            self.n_routed_experts = experts

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeVllmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
            self.moe_layers = [
                self.model.layers[0].mlp.experts,
                self.model.layers[1].mlp.experts,
            ]

    model = FakeVllmModel()
    old_runner_0 = model.moe_layers[0]
    old_runner_1 = model.moe_layers[1]
    original_pack = pack_vllm_routed_experts
    observed_states = []

    def observing_pack(routed_experts, keep_mask, *, output_device=None):
        observed_states.append(tuple(model.moe_layers))
        return original_pack(routed_experts, keep_mask, output_device=output_device)

    monkeypatch.setattr("ramp_moe.vllm_compact.pack_vllm_routed_experts", observing_pack)

    install_compact_qwen3_moe_(model, masks)

    assert len(observed_states) == 2
    assert observed_states[0][0] is None
    assert observed_states[0][1] is old_runner_1
    assert isinstance(observed_states[1][0], CompactVllmQwen3Experts)
    assert observed_states[1][1] is None
    assert model.moe_layers == [
        model.model.layers[0].mlp.experts,
        model.model.layers[1].mlp.experts,
    ]
    assert all(isinstance(layer, CompactVllmQwen3Experts) for layer in model.moe_layers)


def test_install_compact_qwen3_moe_detaches_old_runner_before_upload(monkeypatch):
    torch.manual_seed(181)
    experts, channels, hidden, top_k = 2, 4, 6, 1
    logits = torch.randn(2, experts)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate(logits)
            self.shared_expert = AddOneSharedExpert()
            self.experts = FakeRoutedExperts(
                torch.randn(experts, 2 * channels, hidden),
                torch.randn(experts, hidden, channels),
                top_k=top_k,
            )
            self.n_routed_experts = experts

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeVllmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer()])
            self.moe_layers = [self.model.layers[0].mlp.experts]

    model = FakeVllmModel()
    shared = model.model.layers[0].mlp.shared_expert
    upload_states = []

    def observing_to_device(compact, device):
        upload_states.append((model.model.layers[0].mlp.experts, tuple(model.moe_layers)))
        return compact

    monkeypatch.setattr("ramp_moe.vllm_compact._compact_to_device", observing_to_device)

    install_compact_qwen3_moe_(model, [mask])

    assert upload_states == [(None, (None,))]
    replacement = model.model.layers[0].mlp.experts
    assert isinstance(replacement, CompactVllmQwen3Experts)
    assert replacement.shared_expert is shared


def test_install_compact_qwen3_moe_releases_old_runner_and_w13_before_upload(monkeypatch):
    torch.manual_seed(182)
    experts, channels, hidden, top_k = 2, 4, 6, 1
    logits = torch.randn(2, experts)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate(logits)
            self.experts = FakeRoutedExperts(
                torch.randn(experts, 2 * channels, hidden),
                torch.randn(experts, hidden, channels),
                top_k=top_k,
            )
            self.n_routed_experts = experts

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeVllmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer()])
            self.moe_layers = [self.model.layers[0].mlp.experts]

    model = FakeVllmModel()
    old_runner = model.moe_layers[0]
    old_runner_ref = weakref.ref(old_runner)
    old_w13_ref = weakref.ref(old_runner.w13_weight)
    del old_runner
    observed_refs = []

    def observing_to_device(compact, device):
        observed_refs.append((old_runner_ref(), old_w13_ref()))
        return compact

    monkeypatch.setattr("ramp_moe.vllm_compact._compact_to_device", observing_to_device)

    install_compact_qwen3_moe_(model, [mask])

    assert observed_refs == [(None, None)]


def test_install_compact_qwen3_moe_scrubs_dense_storage_despite_external_refs(monkeypatch):
    torch.manual_seed(183)
    experts, channels, hidden, top_k = 2, 4, 6, 1
    logits = torch.randn(2, experts)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate(logits)
            self.experts = FakeRoutedExperts(
                torch.randn(experts, 2 * channels, hidden),
                torch.randn(experts, hidden, channels),
                top_k=top_k,
            )
            self.n_routed_experts = experts

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeBlock()

    class FakeVllmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FakeLayer()])
            self.moe_layers = [self.model.layers[0].mlp.experts]

    model = FakeVllmModel()
    external_runner_ref = model.moe_layers[0]
    external_w13_ref = external_runner_ref.w13_weight
    external_w2_ref = external_runner_ref.w2_weight
    observed = []

    def observing_to_device(compact, device):
        observed.append(
            (
                external_w13_ref.device.type,
                int(external_w13_ref.numel()),
                external_w2_ref.device.type,
                int(external_w2_ref.numel()),
            )
        )
        return compact

    monkeypatch.setattr("ramp_moe.vllm_compact._compact_to_device", observing_to_device)

    install_compact_qwen3_moe_(model, [mask])

    assert external_runner_ref is not model.model.layers[0].mlp.experts
    assert observed == [("cpu", 0, "cpu", 0)]


def test_compact_qwen3_shared_expert_is_applied_once():
    torch.manual_seed(19)
    experts, channels, hidden, tokens, top_k = 2, 3, 5, 4, 1
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    logits = torch.randn(tokens, experts)
    mask = torch.ones(experts, channels, dtype=torch.bool)
    shared = AddOneSharedExpert()
    compact = pack_vllm_routed_experts(FakeRoutedExperts(w13, w2, top_k), mask)
    module = CompactVllmQwen3Experts(
        compact,
        gate=FakeGate(logits),
        top_k=top_k,
        renormalize=True,
        shared_expert=shared,
    )
    x = torch.randn(tokens, hidden)

    actual = module(hidden_states=x)
    expected = dense_masked_reference(x, logits, w13, w2, mask, top_k) + 1

    assert shared.calls == 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_compact_qwen3_cuda_hot_path_skips_index_validation_by_default(monkeypatch):
    torch.manual_seed(20)
    experts, channels, hidden, tokens, top_k = 2, 3, 5, 4, 1
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    logits = torch.randn(tokens, experts)
    mask = torch.ones(experts, channels, dtype=torch.bool)
    compact = pack_vllm_routed_experts(FakeRoutedExperts(w13, w2, top_k), mask)
    module = CompactVllmQwen3Experts(
        compact,
        gate=FakeGate(logits),
        top_k=top_k,
        renormalize=True,
    )
    observed = []

    def fake_ragged(compact, hidden_states, expert_indices, router_weights, **kwargs):
        observed.append(kwargs["validate_indices"])
        return torch.zeros(
            (hidden_states.shape[0] * kwargs["routes_per_token"], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
        )

    monkeypatch.delenv("RAMP_MOE_VLLM_VALIDATE_INDICES", raising=False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr("ramp_moe.vllm_compact.ragged_expert_forward", fake_ragged)

    output = module(hidden_states=torch.randn(tokens, hidden))

    assert observed == [False]
    assert output.shape == (tokens, hidden)


def test_compact_qwen3_cuda_debug_env_enables_index_validation(monkeypatch):
    torch.manual_seed(21)
    experts, channels, hidden, tokens, top_k = 2, 3, 5, 4, 1
    w13 = torch.randn(experts, 2 * channels, hidden)
    w2 = torch.randn(experts, hidden, channels)
    logits = torch.randn(tokens, experts)
    mask = torch.ones(experts, channels, dtype=torch.bool)
    compact = pack_vllm_routed_experts(FakeRoutedExperts(w13, w2, top_k), mask)
    module = CompactVllmQwen3Experts(
        compact,
        gate=FakeGate(logits),
        top_k=top_k,
        renormalize=True,
    )
    observed = []

    def fake_ragged(compact, hidden_states, expert_indices, router_weights, **kwargs):
        observed.append(kwargs["validate_indices"])
        return torch.zeros(
            (hidden_states.shape[0] * kwargs["routes_per_token"], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
        )

    monkeypatch.setenv("RAMP_MOE_VLLM_VALIDATE_INDICES", "1")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr("ramp_moe.vllm_compact.ragged_expert_forward", fake_ragged)

    output = module(hidden_states=torch.randn(tokens, hidden))

    assert observed == [True]
    assert output.shape == (tokens, hidden)


def test_pack_vllm_routed_experts_rejects_noncanonical_shapes():
    routed = FakeRoutedExperts(
        torch.randn(2, 7, 4),
        torch.randn(2, 4, 3),
    )
    with pytest.raises(ValueError, match="2 \\* channel_count"):
        pack_vllm_routed_experts(routed, torch.ones(2, 3, dtype=torch.bool))


def test_load_masks_from_checkpoint_accepts_repository_schema(tmp_path):
    masks = {
        "10": torch.ones(1, 2, dtype=torch.bool),
        "2": torch.zeros(1, 2, dtype=torch.bool),
    }
    path = tmp_path / "mask.pt"
    torch.save({"model": "local/test", "masks": masks}, path)

    loaded = load_masks_from_checkpoint(path)

    assert len(loaded) == 2
    torch.testing.assert_close(loaded[0], masks["2"])
    torch.testing.assert_close(loaded[1], masks["10"])


def test_pack_vllm_routed_experts_rejects_quantized_and_ep_layouts():
    routed = FakeRoutedExperts(
        torch.randn(2, 6, 4),
        torch.randn(2, 4, 3),
    )
    routed.quant_config = object()
    with pytest.raises(NotImplementedError, match="unquantized"):
        pack_vllm_routed_experts(routed, torch.ones(2, 3, dtype=torch.bool))

    routed.quant_config = None
    routed.register_buffer("expert_map", torch.tensor([0, -1], dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="direct global-to-local"):
        pack_vllm_routed_experts(routed, torch.ones(2, 3, dtype=torch.bool))


def test_register_compact_qwen3_model_when_vllm_available():
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    register_compact_qwen3_model()

    assert COMPACT_QWEN3_ARCH in ModelRegistry.get_supported_archs()
