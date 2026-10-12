from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from scripts.prune_universal_moe import (
    _capture_routed_inputs,
    _load_fixed_stage1_masks,
    _next_output_scale,
    _pop_routed_inputs,
    _profile_spec,
    _rate_distortion_signals,
    _release_replay_buffers_,
    _universal_termination_token_mask,
    _update_universal_statistics,
    parse_args,
)


class _Echo(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_router_and_expert_hooks_preserve_distinct_hidden_states() -> None:
    view = SimpleNamespace(router=_Echo(), experts=_Echo())
    pending, handles = _capture_routed_inputs(view)
    router_hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expert_hidden = router_hidden + 100
    try:
        view.router(router_hidden.reshape(-1, 4))
        view.experts(expert_hidden.reshape(-1, 4))
    finally:
        for handle in handles:
            handle.remove()
    view.router(router_hidden.reshape(-1, 4))

    captured_router, captured_experts = _pop_routed_inputs(
        pending,
        batch_size=2,
        sequence_length=3,
        hidden_size=4,
    )
    torch.testing.assert_close(captured_router, router_hidden)
    torch.testing.assert_close(captured_experts, expert_hidden)


def test_legacy_moe_capture_reuses_shared_block_input() -> None:
    block = _Echo()
    view = SimpleNamespace(
        packed=False,
        mask_module=block,
        router=_Echo(),
        experts=nn.ModuleList([_Echo()]),
    )
    pending, handles = _capture_routed_inputs(view)
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    try:
        block(hidden)
    finally:
        for handle in handles:
            handle.remove()

    captured_router, captured_experts = _pop_routed_inputs(
        pending,
        batch_size=2,
        sequence_length=3,
        hidden_size=4,
    )
    torch.testing.assert_close(captured_router, hidden)
    torch.testing.assert_close(captured_experts, hidden)


def test_routed_input_capture_rejects_missing_expert_call() -> None:
    view = SimpleNamespace(router=_Echo(), experts=_Echo())
    pending, handles = _capture_routed_inputs(view)
    try:
        view.router(torch.ones(6, 4))
    finally:
        for handle in handles:
            handle.remove()

    with pytest.raises(RuntimeError, match="exactly one"):
        _pop_routed_inputs(
            pending,
            batch_size=2,
            sequence_length=3,
            hidden_size=4,
        )


def test_rate_distortion_recapture_releases_stale_replay_buffers() -> None:
    inputs = torch.ones((4, 8, 16))
    outputs = torch.zeros_like(inputs)

    _release_replay_buffers_(inputs, outputs)

    assert inputs.numel() == 0
    assert outputs.numel() == 0
    assert inputs.untyped_storage().nbytes() == 0
    assert outputs.untyped_storage().nbytes() == 0


def test_next_output_scale_handles_qwen35_zero_centered_norm() -> None:
    class Qwen3_5MoeRMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

    class Layer(nn.Module):
        def __init__(self, norm: nn.Module) -> None:
            super().__init__()
            self.input_layernorm = norm

    first = Layer(nn.LayerNorm(4))
    second = Layer(Qwen3_5MoeRMSNorm())
    model = SimpleNamespace(model=SimpleNamespace(norm=nn.LayerNorm(4)))

    torch.testing.assert_close(
        _next_output_scale(model, [first, second], 0), torch.ones(4)
    )
    assert _next_output_scale(model, [first, second], 1) is model.model.norm.weight


def test_smoke_default_covers_every_hard_math_level_type(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--competition-math-source",
            "/math.parquet",
            "--competition-math-exclusion-manifest",
            "/exclude.json",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
            "--smoke",
        ],
    )
    assert parse_args().smoke_nsamples == 49

    monkeypatch.setattr("sys.argv", [*sys.argv, "--smoke-nsamples", "28"])
    with pytest.raises(SystemExit):
        parse_args()


def test_review_sensitivity_parameters_are_explicit_and_validated(monkeypatch) -> None:
    base = [
        "prune_universal_moe.py",
        "--profile",
        "review_stage1_envelope",
        "--model-source",
        "/model",
        "--model-name",
        "model",
        "--competition-math-source",
        "/math.parquet",
        "--competition-math-exclusion-manifest",
        "/exclude.json",
        "--checkpoint",
        "/mask.pt",
        "--output",
        "/result.json",
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            *base,
            "--camera-alpha",
            "0.7",
            "--anchor-general-weight",
            "0.8",
            "--trust-radius-divisor",
            "16",
            "--interaction-strength-override",
            "0.4",
        ],
    )
    args = parse_args()
    assert args.camera_alpha == 0.7
    assert args.anchor_general_weight == 0.8
    assert args.trust_radius_divisor == 16
    assert args.interaction_strength_override == 0.4

    monkeypatch.setattr("sys.argv", [*base, "--trust-radius-divisor", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_fixed_stage1_checkpoint_loader_validates_and_returns_exact_masks(
    tmp_path,
) -> None:
    path = tmp_path / "stage1.pt"
    masks = [torch.tensor([[True, False], [False, True]]) for _ in range(3)]
    torch.save(
        {
            "model": "model",
            "calibration_aggregation": "universal_capability_envelope",
            "max_swaps": 0,
            "masks": masks,
        },
        path,
    )

    loaded, digest = _load_fixed_stage1_masks(
        path,
        model_name="model",
        layer_count=3,
        expert_count=2,
        channel_count=2,
        layer_limit=2,
    )

    assert len(loaded) == 2
    assert all(torch.equal(left, right) for left, right in zip(loaded, masks))
    assert len(digest) == 64

    with pytest.raises(ValueError, match="model mismatch"):
        _load_fixed_stage1_masks(
            path,
            model_name="other",
            layer_count=3,
            expert_count=2,
            channel_count=2,
            layer_limit=2,
        )


def test_camera_profile_uses_standard_wikitext_uniform_exact50() -> None:
    spec = _profile_spec(
        "camera",
        layer_count=3,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )

    assert spec.calibration_corpus == "wikitext"
    assert spec.nsamples == 128
    assert spec.sequence_length == 2048
    assert spec.reservoir_size == 512
    assert spec.method == "camera"
    assert spec.calibration_aggregation == "pooled"
    assert spec.layer_budget_schedule == "uniform_exact50"
    assert spec.layer_budgets == (16, 16, 16)


@pytest.mark.parametrize(
    (
        "profile",
        "calibration_corpus",
        "calibration_aggregation",
        "method",
        "max_swaps",
        "domain_names",
    ),
    [
        (
            "review_camera_single_448",
            "wikitext",
            "single_domain",
            "camera",
            0,
            ("wikitext",),
        ),
        (
            "review_camera_four_domain_pooled",
            "reasoning_hardmath_code_append",
            "four_domain_raw_pooled",
            "camera",
            0,
            ("pooled_four_domain",),
        ),
        (
            "review_camera_four_domain_pooled_ramp",
            "reasoning_hardmath_code_append",
            "four_domain_raw_pooled",
            "ramp_camera",
            1,
            ("pooled_four_domain",),
        ),
        (
            "review_camera_four_domain_normalized",
            "reasoning_hardmath_code_append",
            "four_domain_mean_normalized_equal_weight",
            "camera",
            0,
            ("wikitext", "reasoning", "hard_math", "code"),
        ),
        (
            "review_anchor_only",
            "reasoning_hardmath_code_append",
            "universal_anchor_only",
            "camera",
            0,
            ("wikitext", "reasoning", "hard_math", "code"),
        ),
        (
            "review_stage1_envelope",
            "reasoning_hardmath_code_append",
            "universal_capability_envelope",
            "ramp_camera",
            0,
            ("wikitext", "reasoning", "hard_math", "code"),
        ),
        (
            "review_stage1_ramp",
            "reasoning_hardmath_code_append",
            "universal_capability_envelope",
            "ramp_camera",
            1,
            ("wikitext", "reasoning", "hard_math", "code"),
        ),
    ],
)
def test_review_v1_fair_ablation_profiles_share_exact50_tail_budget(
    profile: str,
    calibration_corpus: str,
    calibration_aggregation: str,
    method: str,
    max_swaps: int,
    domain_names: tuple[str, ...],
) -> None:
    spec = _profile_spec(
        profile,
        layer_count=6,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )

    assert spec.calibration_corpus == calibration_corpus
    assert spec.calibration_aggregation == calibration_aggregation
    assert spec.nsamples == 448
    assert spec.sequence_length == 2048
    assert spec.reservoir_size == 512
    assert spec.method == method
    assert spec.layer_budget_schedule == "tail_protected_normalized_exact50"
    assert spec.layer_budgets == (14, 14, 14, 14, 20, 20)
    assert sum(spec.layer_budgets) * 2 == 6 * 4 * 8
    assert spec.interaction_strength == (0.2 if method == "ramp_camera" else None)
    assert spec.max_swaps == max_swaps
    assert spec.domain_names == domain_names
    assert spec.rate_distortion_capacity_signal is None
    assert spec.rate_distortion_budget_scope is None
    assert spec.rate_distortion_score_mode is None
    assert spec.rate_distortion_max_width_delta is None


def test_other_review_v1_fair_ablation_profiles_reject_non_exact50_endpoint() -> None:
    with pytest.raises(ValueError, match="frozen at exact50"):
        _profile_spec(
            "review_camera_four_domain_normalized",
            layer_count=6,
            expert_count=4,
            channel_count=8,
            pruning_rate=0.25,
            smoke=False,
        )


def test_review_anchor_only_supports_exact75_with_stage1_budget() -> None:
    anchor = _profile_spec(
        "review_anchor_only",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )
    stage1 = _profile_spec(
        "review_stage1_envelope",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )

    assert anchor.layer_budget_schedule == stage1.layer_budget_schedule
    assert anchor.layer_budgets == stage1.layer_budgets
    assert anchor.calibration_corpus == stage1.calibration_corpus
    assert anchor.domain_names == stage1.domain_names
    assert anchor.calibration_aggregation == "universal_anchor_only"
    assert anchor.max_swaps == 0


def test_review_stage1_envelope_supports_exact75_endpoint() -> None:
    spec = _profile_spec(
        "review_stage1_envelope",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )

    assert spec.layer_budget_schedule == "tail_protected_normalized_exact75"
    assert spec.layer_budgets == (21, 21, 21, 21, 30, 30)
    assert sum(spec.layer_budgets) == 6 * 4 * 8 * 3 // 4
    assert spec.calibration_aggregation == "universal_capability_envelope"
    assert spec.max_swaps == 0


def test_fixed_width_review_stage1_ramp_supports_exact75_endpoint() -> None:
    stage1 = _profile_spec(
        "review_stage1_envelope",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )
    ramp = _profile_spec(
        "review_stage1_ramp",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )

    assert ramp.layer_budget_schedule == stage1.layer_budget_schedule
    assert ramp.layer_budgets == stage1.layer_budgets
    assert ramp.calibration_aggregation == stage1.calibration_aggregation
    assert ramp.max_swaps == 1


def test_review_stage1_ramp_exact75_requires_fixed_stage1_checkpoint(monkeypatch) -> None:
    argv = [
        "prune_universal_moe.py",
        "--profile",
        "review_stage1_ramp",
        "--pruning-rate",
        "0.25",
        "--model-source",
        "/model",
        "--model-name",
        "model",
        "--checkpoint",
        "/mask.pt",
        "--output",
        "/result.json",
        "--competition-math-source",
        "/math.parquet",
        "--competition-math-exclusion-manifest",
        "/exclusions.json",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        "sys.argv", [*argv, "--fixed-stage1-checkpoint", "/stage1.pt"]
    )
    assert parse_args().pruning_rate == 0.25


def test_review_v1_uniform_ablation_changes_only_the_layer_schedule() -> None:
    tail = _profile_spec(
        "review_stage1_ramp",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )
    uniform = _profile_spec(
        "review_stage1_ramp_uniform",
        layer_count=6,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )

    assert uniform.layer_budget_schedule == "uniform_normalized_exact50"
    assert uniform.layer_budgets == (16, 16, 16, 16, 16, 16)
    assert sum(uniform.layer_budgets) == sum(tail.layer_budgets)
    assert uniform.calibration_aggregation == tail.calibration_aggregation
    assert uniform.method == tail.method
    assert uniform.max_swaps == tail.max_swaps == 1
    assert uniform.nsamples == tail.nsamples == 448


@pytest.mark.parametrize(
    ("layer_count", "expert_count", "channel_count", "retained_total"),
    [
        (30, 128, 704, 1_351_680),
        (40, 256, 512, 2_621_440),
    ],
)
def test_universal_termination_profile_freezes_idea058_append448_exact50(
    layer_count: int,
    expert_count: int,
    channel_count: int,
    retained_total: int,
) -> None:
    spec = _profile_spec(
        "universal_termination",
        layer_count=layer_count,
        expert_count=expert_count,
        channel_count=channel_count,
        smoke=False,
    )

    assert spec.calibration_corpus == "reasoning_hardmath_code_append"
    assert spec.calibration_aggregation == (
        "universal_termination_capacity_envelope"
    )
    assert spec.nsamples == 448
    assert spec.sequence_length == 2048
    assert spec.reservoir_size == 512
    assert spec.method == "ramp_camera"
    assert spec.layer_budget_schedule == "tail_protected_normalized_exact50"
    assert sum(spec.layer_budgets) == retained_total
    assert retained_total * 2 == layer_count * expert_count * channel_count
    assert spec.interaction_strength == 0.2
    assert spec.max_swaps == 1
    assert spec.domain_names == (
        "wikitext",
        "reasoning",
        "hard_math",
        "code",
        "assistant_end",
    )


@pytest.mark.parametrize(
    ("layer_count", "expert_count", "channel_count", "retained_total"),
    [
        (30, 128, 704, 2_027_520),
        (40, 256, 512, 3_932_160),
    ],
)
def test_universal_termination_profile_supports_exact25_pruning_endpoint(
    layer_count: int,
    expert_count: int,
    channel_count: int,
    retained_total: int,
) -> None:
    first = _profile_spec(
        "universal_termination",
        layer_count=layer_count,
        expert_count=expert_count,
        channel_count=channel_count,
        pruning_rate=0.25,
        smoke=False,
    )
    second = _profile_spec(
        "universal_termination",
        layer_count=layer_count,
        expert_count=expert_count,
        channel_count=channel_count,
        pruning_rate=0.25,
        smoke=False,
    )

    assert first.layer_budget_schedule == "tail_protected_normalized_exact75"
    assert first.layer_budgets == second.layer_budgets
    assert sum(first.layer_budgets) == retained_total
    assert retained_total * 4 == 3 * layer_count * expert_count * channel_count
    tail_start = 2 * layer_count // 3
    assert max(first.layer_budgets[:tail_start]) < min(
        first.layer_budgets[tail_start:]
    )


@pytest.mark.parametrize(
    "profile", ("universal", "universal_termination", "camera", "rate_distortion")
)
def test_explicit_half_pruning_is_pointwise_compatible_with_legacy_profile(
    profile: str,
) -> None:
    legacy = _profile_spec(
        profile,
        layer_count=40,
        expert_count=256,
        channel_count=512,
        smoke=False,
    )
    explicit = _profile_spec(
        profile,
        layer_count=40,
        expert_count=256,
        channel_count=512,
        pruning_rate=0.50,
        smoke=False,
    )

    assert explicit == legacy
    assert explicit.layer_budget_schedule.endswith("exact50")


@pytest.mark.parametrize(
    ("profile", "expected_schedule"),
    [
        ("camera", "uniform_exact75"),
        ("rate_distortion", "layer_expert_rate_distortion_exact75"),
    ],
)
def test_non_tail_profiles_preserve_exact25_pruning_endpoint(
    profile: str, expected_schedule: str
) -> None:
    spec = _profile_spec(
        profile,
        layer_count=3,
        expert_count=4,
        channel_count=8,
        pruning_rate=0.25,
        smoke=False,
    )

    assert spec.layer_budget_schedule == expected_schedule
    assert spec.layer_budgets == (24, 24, 24)


def test_universal_termination_mask_excludes_general_and_incomplete_spans() -> None:
    input_ids = torch.tensor(
        [
            [7, 8, 1, 9, 10, 0],
            [7, 8, 2, 9, 10, 0],
            [9, 10, 7, 8, 3, 0],
            [7, 8, 4, 9, 10, 0],
            [7, 8, 5, 9, 10, 0],
        ]
    )
    calibration = {
        "assistant_marker_token_ids": [7, 8],
        "assistant_end_marker_token_ids": [9, 10],
    }

    mask = _universal_termination_token_mask(
        input_ids,
        calibration,
        general_count=1,
        reasoning_count=1,
        hard_math_count=2,
    )

    assert not bool(mask[0].any().item())
    assert mask[1].tolist() == [False, False, True, True, False, False]
    assert not bool(mask[2].any().item())
    assert bool(mask[3].any().item())
    assert bool(mask[4].any().item())


def test_universal_statistics_routes_end_support_only_to_capacity_domain() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

        def update(
            self,
            router: torch.Tensor,
            *,
            expert_hidden_states: torch.Tensor,
            token_selection_mask: torch.Tensor | None = None,
        ) -> None:
            self.calls.append(
                (router.clone(), expert_hidden_states.clone(), token_selection_mask)
            )

    domains = tuple(Recorder() for _ in range(4))
    termination = Recorder()
    router = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    experts = router + 1_000
    termination_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, False, True],
        ]
    )

    _update_universal_statistics(
        router,
        experts,
        domain_accumulators=domains,
        domain_counts=(1, 1, 2, 1),
        batch_start=0,
        termination_accumulator=termination,
        termination_token_mask=termination_mask,
    )

    assert all(len(domain.calls) == 1 for domain in domains)
    assert all(domain.calls[0][2] is None for domain in domains)
    termination_router, termination_experts, selected = termination.calls[0]
    torch.testing.assert_close(termination_router, router[1:])
    torch.testing.assert_close(termination_experts, experts[1:])
    assert torch.equal(selected, termination_mask[1:])


def test_rate_distortion_profile_uses_nominal_half_width_only_as_anchor() -> None:
    spec = _profile_spec(
        "rate_distortion",
        layer_count=3,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )

    assert spec.calibration_corpus == "reasoning_hardmath_code_append"
    assert spec.calibration_aggregation == "universal_capability_envelope"
    assert spec.layer_budget_schedule == "layer_expert_rate_distortion_exact50"
    assert spec.layer_budgets == (16, 16, 16)
    assert spec.method == "ramp_camera"
    assert spec.rate_distortion_capacity_signal == "capability_envelope"
    assert spec.rate_distortion_budget_scope == "model"
    assert spec.rate_distortion_score_mode == "lamp"
    assert spec.rate_distortion_max_width_delta == 1


@pytest.mark.parametrize(
    (
        "profile_name",
        "max_swaps",
        "capacity_signal",
        "budget_scope",
        "score_mode",
        "max_width_delta",
    ),
    [
        (
            "rate_distortion_no_ramp",
            0,
            "capability_envelope",
            "model",
            "lamp",
            1,
        ),
        ("rate_distortion_anchor_only", 1, "anchor", "model", "lamp", 1),
        (
            "rate_distortion_uniform_layer",
            1,
            "capability_envelope",
            "layer",
            "lamp",
            1,
        ),
        (
            "rate_distortion_raw_energy",
            1,
            "capability_envelope",
            "model",
            "raw",
            1,
        ),
        (
            "rate_distortion_fixed_expert_width",
            1,
            "capability_envelope",
            "model",
            "lamp",
            0,
        ),
    ],
)
def test_rate_distortion_ablation_profiles_change_exactly_one_component(
    profile_name: str,
    max_swaps: int,
    capacity_signal: str,
    budget_scope: str,
    score_mode: str,
    max_width_delta: int,
) -> None:
    spec = _profile_spec(
        profile_name,
        layer_count=3,
        expert_count=4,
        channel_count=8,
        smoke=False,
    )

    assert spec.calibration_corpus == "reasoning_hardmath_code_append"
    assert spec.calibration_aggregation == "universal_capability_envelope"
    assert spec.layer_budgets == (16, 16, 16)
    assert spec.max_swaps == max_swaps
    assert spec.rate_distortion_capacity_signal == capacity_signal
    assert spec.rate_distortion_budget_scope == budget_scope
    assert spec.rate_distortion_score_mode == score_mode
    assert spec.rate_distortion_max_width_delta == max_width_delta


def test_rate_distortion_accepts_small_but_positive_domain_importance() -> None:
    class Domain:
        routed_token_counts = torch.ones(2, dtype=torch.long)

        def camera_importance(self, _alpha: float) -> torch.Tensor:
            return torch.full((2, 4), 1e-8)

    class Accumulator:
        domains = (Domain(), Domain(), Domain(), Domain())

        def camera_importance(self, _alpha: float) -> torch.Tensor:
            return torch.arange(1, 9, dtype=torch.float32).reshape(2, 4) * 1e-8

        def allocation_importance_for_reference_widths(
            self, _alpha: float, _widths: torch.Tensor
        ) -> torch.Tensor:
            return self.camera_importance(_alpha)

    anchor, allocation = _rate_distortion_signals(
        Accumulator(), camera_alpha=0.9, capacity_signal="capability_envelope"
    )

    assert torch.isfinite(anchor).all()
    assert torch.equal(allocation, anchor)


def test_rate_distortion_reference_width_uses_selected_pruning_endpoint() -> None:
    class Domain:
        routed_token_counts = torch.ones(2, dtype=torch.long)

        def camera_importance(self, _alpha: float) -> torch.Tensor:
            return torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

    class Accumulator:
        domains = (Domain(), Domain(), Domain(), Domain())
        reference_widths: torch.Tensor | None = None

        def camera_importance(self, _alpha: float) -> torch.Tensor:
            return self.domains[0].camera_importance(_alpha)

        def allocation_importance_for_reference_widths(
            self, _alpha: float, widths: torch.Tensor
        ) -> torch.Tensor:
            self.reference_widths = widths
            return self.camera_importance(_alpha)

    accumulator = Accumulator()
    _rate_distortion_signals(
        accumulator,
        camera_alpha=0.9,
        capacity_signal="capability_envelope",
        pruning_rate=0.25,
    )

    assert accumulator.reference_widths is not None
    assert int(accumulator.reference_widths.sum().item()) == 6


def test_camera_profile_cli_does_not_require_hard_math_artifacts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--profile",
            "camera",
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
            "--smoke",
            "--smoke-nsamples",
            "1",
        ],
    )

    args = parse_args()

    assert args.profile == "camera"
    assert args.competition_math_source is None
    assert args.competition_math_exclusion_manifest is None


def test_review_camera_single_profile_cli_does_not_require_hard_math_artifacts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--profile",
            "review_camera_single_448",
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
            "--smoke",
            "--smoke-nsamples",
            "1",
        ],
    )

    args = parse_args()

    assert args.profile == "review_camera_single_448"
    assert args.competition_math_source is None
    assert args.competition_math_exclusion_manifest is None


@pytest.mark.parametrize(
    "profile",
    [
        "review_camera_four_domain_pooled",
        "review_camera_four_domain_pooled_ramp",
        "review_camera_four_domain_normalized",
        "review_anchor_only",
        "review_stage1_envelope",
        "review_stage1_ramp",
        "review_stage1_ramp_uniform",
    ],
)
def test_review_four_domain_profiles_require_hard_math_artifacts(
    monkeypatch, profile: str
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--profile",
            profile,
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_universal_profile_still_requires_hard_math_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_universal_termination_profile_is_a_frozen_cli_choice(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_universal_moe.py",
            "--profile",
            "universal_termination",
            "--model-source",
            "/model",
            "--model-name",
            "model",
            "--competition-math-source",
            "/math.parquet",
            "--competition-math-exclusion-manifest",
            "/exclude.json",
            "--checkpoint",
            "/mask.pt",
            "--output",
            "/result.json",
        ],
    )

    args = parse_args()

    assert args.profile == "universal_termination"
    assert args.pruning_rate == 0.50


@pytest.mark.parametrize("pruning_rate", ("0.25", "0.50"))
def test_cli_accepts_only_frozen_pruning_rate_endpoints(
    monkeypatch, pruning_rate: str
) -> None:
    argv = [
        "prune_universal_moe.py",
        "--profile",
        "camera",
        "--pruning-rate",
        pruning_rate,
        "--model-source",
        "/model",
        "--model-name",
        "model",
        "--checkpoint",
        "/mask.pt",
        "--output",
        "/result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert parse_args().pruning_rate == float(pruning_rate)

    argv[argv.index(pruning_rate)] = "0.30"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()
