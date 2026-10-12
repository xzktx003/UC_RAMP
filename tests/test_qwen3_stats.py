import pytest
import torch
from torch import nn

from ramp_moe.baselines.camera import camera_microexpert_importance
from ramp_moe.core import (
    build_centered_difference_interaction_gram,
    build_energy_calibrated_difference_interaction_gram,
    build_interaction_gram,
)
from ramp_moe.qwen3_stats import (
    CapabilityComplementStatsAccumulator,
    DualSpecialistCapacityEnvelopeStatsAccumulator,
    CapabilityEnvelopeStatsAccumulator,
    CapabilityUnionStatsAccumulator,
    DomainRobustStatsAccumulator,
    MultiDomainRobustStatsAccumulator,
    PhaseCapacityEnvelopeStatsAccumulator,
    Qwen3LayerStatsAccumulator,
    router_weighted_phase_means,
    UniversalEqualDomainStatsAccumulator,
    UniversalCapabilityEnvelopeStatsAccumulator,
    UniversalTerminationCapacityEnvelopeStatsAccumulator,
    combine_domain_importance,
)


def test_domain_robust_importance_protects_reasoning_specific_channel() -> None:
    general = torch.tensor([[100.0, 99.0, 1.0, 1.0]])
    reasoning = torch.tensor([[1.0, 1.0, 100.0, 1.0]])

    combined = combine_domain_importance(general, reasoning)

    assert set(torch.topk(combined.flatten(), k=2).indices.tolist()) == {0, 2}
    rescaled = combine_domain_importance(general, reasoning * 1000)
    assert torch.equal(
        torch.argsort(combined.flatten(), descending=True),
        torch.argsort(rescaled.flatten(), descending=True),
    )


def test_domain_anchored_importance_limits_reasoning_domain_displacement() -> None:
    general = torch.tensor([[100.0, 99.0, 1.0, 1.0]])
    reasoning = torch.tensor([[1.0, 1.0, 100.0, 1.0]])

    combined = combine_domain_importance(
        general, reasoning, aggregation="domain_anchored"
    )

    assert set(torch.topk(combined.flatten(), k=2).indices.tolist()) == {0, 1}


class _FakeDomainAccumulator:
    expert_count = 1
    channel_count = 2
    device = torch.device("cpu")

    def __init__(
        self,
        importance: torch.Tensor,
        gram: torch.Tensor,
        routed: int,
        difference_importance: torch.Tensor | None = None,
    ) -> None:
        self.expert_count = importance.shape[0]
        self.channel_count = importance.shape[1]
        self._importance = importance
        self._difference_importance = (
            importance if difference_importance is None else difference_importance
        )
        self._gram = gram
        self.routed_token_counts = torch.tensor([routed])

    def camera_importance(self, _alpha: float) -> torch.Tensor:
        return self._importance

    def difference_importance(self) -> torch.Tensor:
        return self._difference_importance

    def interaction_gram(self, _expert_index: int) -> torch.Tensor:
        return self._gram

    def reconstruction_gram(self, _expert_index: int) -> torch.Tensor:
        return self._gram

    def reservoir_size_for(self, _expert_index: int) -> int:
        return 3


def test_domain_robust_accumulator_balances_grams_and_sums_counts() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(general, reasoning)

    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.0, 4.0]))
    )
    torch.testing.assert_close(
        combined.reconstruction_gram(0), torch.diag(torch.tensor([3.0, 4.0]))
    )
    assert combined.routed_token_counts.tolist() == [12]
    assert combined.reservoir_size_for(0) == 6


def test_domain_anchored_accumulator_uses_ninety_ten_gram_mix() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="domain_anchored"
    )

    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.4]))
    )


def test_hierarchical_robust_decouples_expert_budget_from_channel_importance() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="hierarchical_robust"
    )

    torch.testing.assert_close(combined.allocation_importance(0.9), general._importance)
    torch.testing.assert_close(
        combined.camera_importance(0.9),
        combine_domain_importance(general._importance, reasoning._importance),
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.0, 4.0]))
    )


def test_bounded_hierarchical_robust_exposes_general_reference_and_robust_target() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="bounded_hierarchical_robust"
    )

    torch.testing.assert_close(
        combined.allocation_importance(0.9),
        combine_domain_importance(general._importance, reasoning._importance),
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), general._importance
    )


def test_bounded_domain_anchored_uses_smooth_target_and_general_width_reference() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="bounded_domain_anchored"
    )

    expected = combine_domain_importance(
        general._importance,
        reasoning._importance,
        aggregation="domain_anchored",
    )
    torch.testing.assert_close(combined.camera_importance(0.9), expected)
    torch.testing.assert_close(combined.allocation_importance(0.9), expected)
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), general._importance
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.4]))
    )


def test_anchored_capacity_trust_region_separates_width_and_channel_signals() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="anchored_capacity_trust_region"
    )
    anchored = combine_domain_importance(
        general._importance,
        reasoning._importance,
        aggregation="domain_anchored",
    )

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(combined.allocation_importance(0.9), general._importance)
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.4]))
    )


def test_anchored_robust_capacity_migrates_widths_without_robust_channel_spikes() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="anchored_robust_capacity"
    )
    anchored = combine_domain_importance(
        general._importance,
        reasoning._importance,
        aggregation="domain_anchored",
    )
    robust = combine_domain_importance(general._importance, reasoning._importance)

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(combined.allocation_importance(0.9), robust)
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.4]))
    )


def test_anchored_geometric_capacity_uses_parameter_free_pareto_midpoint() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="anchored_geometric_capacity"
    )
    anchored = combine_domain_importance(
        general._importance,
        reasoning._importance,
        aggregation="domain_anchored",
    )
    robust = combine_domain_importance(general._importance, reasoning._importance)

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(
        combined.allocation_importance(0.9), torch.sqrt(anchored * robust)
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )


def test_anchored_difference_capacity_uses_difference_only_for_widths() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0])),
        7,
        difference_importance=torch.tensor([[1.0, 16.0]]),
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0]]),
        torch.diag(torch.tensor([2.0, 6.0])),
        5,
        difference_importance=torch.tensor([[9.0, 4.0]]),
    )

    combined = DomainRobustStatsAccumulator(
        general, reasoning, aggregation="anchored_difference_capacity"
    )
    anchored = combine_domain_importance(
        general._importance,
        reasoning._importance,
        aggregation="domain_anchored",
    )
    robust = combine_domain_importance(general._importance, reasoning._importance)
    robust_difference = combine_domain_importance(
        general._difference_importance,
        reasoning._difference_importance,
    )

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(
        combined.allocation_importance(0.9),
        torch.sqrt(robust * robust_difference),
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.4]))
    )


def test_capability_envelope_separates_math_and_code_capacity_spikes() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0, 2.0])),
        7,
    )
    math = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 6.0, 2.0])),
        5,
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 9.0]]),
        torch.diag(torch.tensor([2.0, 2.0, 8.0])),
        3,
    )

    combined = CapabilityEnvelopeStatsAccumulator(general, math, code)
    normalized_general = general._importance / general._importance.mean()
    normalized_math = math._importance / math._importance.mean()
    normalized_code = code._importance / code._importance.mean()
    anchored = (
        0.9 * normalized_general
        + 0.05 * normalized_math
        + 0.05 * normalized_code
    ) * general._importance.mean()
    reference_widths = torch.tensor([2])
    anchor_order = torch.argsort(anchored, dim=1, descending=True)
    evidence_mask = torch.zeros_like(anchored, dtype=torch.bool)
    evidence_mask.scatter_(1, anchor_order[:, :2], True)
    robust_mass = torch.stack(
        [
            (domain * evidence_mask).sum(dim=1)
            for domain in (
                normalized_general * general._importance.mean(),
                normalized_math * general._importance.mean(),
                normalized_code * general._importance.mean(),
            )
        ]
    ).amax(dim=0)
    multiplier = torch.sqrt(
        robust_mass / (anchored * evidence_mask).sum(dim=1)
    )

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(
        combined.allocation_importance_for_reference_widths(
            0.9, reference_widths
        ),
        anchored * multiplier.unsqueeze(1),
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )
    torch.testing.assert_close(
        combined.interaction_gram(0), torch.diag(torch.tensor([3.8, 2.2, 2.3]))
    )
    assert combined.routed_token_counts.tolist() == [15]
    assert combined.reservoir_size_for(0) == 9


def test_universal_capability_envelope_balances_three_specialists_under_one_anchor() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0, 1.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0, 2.0, 2.0])),
        7,
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0, 1.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 6.0, 2.0, 2.0])),
        5,
    )
    hard_math = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 9.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 2.0, 8.0, 2.0])),
        3,
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 1.0, 9.0]]),
        torch.diag(torch.tensor([2.0, 2.0, 2.0, 10.0])),
        1,
    )

    combined = UniversalCapabilityEnvelopeStatsAccumulator(
        general, reasoning, hard_math, code
    )
    domains = (general, reasoning, hard_math, code)
    scaled = tuple(
        domain._importance / domain._importance.mean()
        * general._importance.mean()
        for domain in domains
    )
    specialist_weight = 0.1 / 3
    anchored = 0.9 * scaled[0] + specialist_weight * sum(scaled[1:])
    reference_widths = torch.tensor([2])
    anchor_order = torch.argsort(anchored, dim=1, descending=True)
    evidence_mask = torch.zeros_like(anchored, dtype=torch.bool)
    evidence_mask.scatter_(1, anchor_order[:, :2], True)
    robust_mass = torch.stack(
        [(domain * evidence_mask).sum(dim=1) for domain in scaled]
    ).amax(dim=0)
    multiplier = torch.sqrt(
        robust_mass / (anchored * evidence_mask).sum(dim=1)
    )
    expected_gram = 0.9 * general._gram + specialist_weight * (
        reasoning._gram + hard_math._gram + code._gram
    )

    torch.testing.assert_close(combined.camera_importance(0.9), anchored)
    torch.testing.assert_close(
        combined.allocation_importance_for_reference_widths(
            0.9, reference_widths
        ),
        anchored * multiplier.unsqueeze(1),
    )
    torch.testing.assert_close(
        combined.capacity_multipliers_for_reference_widths(
            0.9, reference_widths
        ),
        multiplier,
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), anchored
    )
    torch.testing.assert_close(combined.interaction_gram(0), expected_gram)
    torch.testing.assert_close(combined.reconstruction_gram(0), expected_gram)
    assert combined.routed_token_counts.tolist() == [16]
    assert combined.reservoir_size_for(0) == 12


def test_universal_capability_envelope_exposes_anchor_weight_for_sensitivity() -> None:
    domains = tuple(
        _FakeDomainAccumulator(
            importance,
            torch.diag(torch.tensor([float(index + 1), float(index + 2)])),
            index + 1,
        )
        for index, importance in enumerate(
            (
                torch.tensor([[9.0, 1.0]]),
                torch.tensor([[1.0, 9.0]]),
                torch.tensor([[2.0, 8.0]]),
                torch.tensor([[3.0, 7.0]]),
            )
        )
    )
    combined = UniversalCapabilityEnvelopeStatsAccumulator(
        *domains, general_weight=0.6
    )
    scaled = tuple(domain._importance for domain in domains)
    expected_anchor = 0.6 * scaled[0] + (0.4 / 3) * sum(scaled[1:])
    expected_gram = 0.6 * domains[0]._gram + (0.4 / 3) * sum(
        domain._gram for domain in domains[1:]
    )

    torch.testing.assert_close(combined.camera_importance(0.9), expected_anchor)
    torch.testing.assert_close(combined.interaction_gram(0), expected_gram)
    assert combined.general_weight == 0.6

    with pytest.raises(ValueError, match="anchor weight"):
        UniversalCapabilityEnvelopeStatsAccumulator(*domains, general_weight=1.1)


def test_universal_equal_domain_accumulator_normalizes_and_weights_domains_equally() -> None:
    domains = (
        _FakeDomainAccumulator(
            torch.tensor([[8.0, 2.0]]), torch.diag(torch.tensor([4.0, 2.0])), 7
        ),
        _FakeDomainAccumulator(
            torch.tensor([[20.0, 80.0]]), torch.diag(torch.tensor([2.0, 6.0])), 5
        ),
        _FakeDomainAccumulator(
            torch.tensor([[6.0, 4.0]]), torch.diag(torch.tensor([8.0, 2.0])), 3
        ),
        _FakeDomainAccumulator(
            torch.tensor([[40.0, 60.0]]), torch.diag(torch.tensor([2.0, 10.0])), 1
        ),
    )

    combined = UniversalEqualDomainStatsAccumulator(*domains)

    values = tuple(domain._importance for domain in domains)
    scales = torch.stack([value.mean() for value in values])
    reference_scale = scales.prod().pow(1 / len(scales))
    expected = sum(
        value / scale * reference_scale
        for value, scale in zip(values, scales)
    ) / len(values)
    expected_gram = sum(domain._gram for domain in domains) / len(domains)

    torch.testing.assert_close(combined.camera_importance(0.9), expected)
    torch.testing.assert_close(combined.allocation_importance(0.9), expected)
    assert combined.allocation_reference_importance(0.9) is None
    torch.testing.assert_close(combined.interaction_gram(0), expected_gram)
    torch.testing.assert_close(combined.reconstruction_gram(0), expected_gram)
    assert combined.routed_token_counts.tolist() == [16]
    assert combined.reservoir_size_for(0) == 12


def test_universal_capability_envelope_accepts_small_positive_domain_scale() -> None:
    importance = torch.tensor([[1.0e-8, 2.0e-8, 3.0e-8]])
    gram = torch.eye(3)
    domains = [
        _FakeDomainAccumulator(importance * scale, gram, routed)
        for scale, routed in zip((1.0, 2.0, 3.0, 4.0), (7, 5, 3, 1))
    ]
    combined = UniversalCapabilityEnvelopeStatsAccumulator(*domains)

    anchor = combined.camera_importance(0.9)
    allocation = combined.allocation_importance_for_reference_widths(
        0.9, torch.tensor([2])
    )

    assert torch.isfinite(anchor).all()
    assert torch.isfinite(allocation).all()
    assert bool((anchor > 0).any().item())


def test_universal_termination_envelope_changes_only_expert_capacity() -> None:
    gram = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 2.0, 1.0], [6.0, 3.0, 1.0]]), gram, 7
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[8.0, 3.0, 1.0], [5.0, 4.0, 1.0]]), gram * 2, 5
    )
    hard_math = _FakeDomainAccumulator(
        torch.tensor([[7.0, 4.0, 1.0], [4.0, 5.0, 1.0]]), gram * 3, 3
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[6.0, 5.0, 1.0], [3.0, 6.0, 1.0]]), gram * 4, 2
    )
    termination = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0]]), gram * 100, 1
    )
    base = UniversalCapabilityEnvelopeStatsAccumulator(
        general, reasoning, hard_math, code
    )
    combined = UniversalTerminationCapacityEnvelopeStatsAccumulator(
        general, reasoning, hard_math, code, termination
    )
    widths = torch.tensor([2, 2])

    base_importance = base.camera_importance(0.9)
    target = combined.allocation_importance_for_reference_widths(0.9, widths)
    ratio = target / base_importance

    torch.testing.assert_close(combined.camera_importance(0.9), base_importance)
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), base_importance
    )
    torch.testing.assert_close(combined.interaction_gram(0), base.interaction_gram(0))
    torch.testing.assert_close(
        combined.reconstruction_gram(0), base.reconstruction_gram(0)
    )
    torch.testing.assert_close(ratio[:, :1], ratio[:, 1:2])
    torch.testing.assert_close(ratio[:, :1], ratio[:, 2:3])
    assert ratio[1, 0] > ratio[0, 0]
    assert combined.termination_routed_token_counts.tolist() == [1]


def test_universal_termination_envelope_rejects_empty_end_evidence() -> None:
    importance = torch.tensor([[3.0, 2.0, 1.0]])
    gram = torch.eye(3)
    domains = [
        _FakeDomainAccumulator(importance, gram, routed)
        for routed in (7, 5, 3, 2)
    ]
    termination = _FakeDomainAccumulator(torch.zeros_like(importance), gram, 0)
    combined = UniversalTerminationCapacityEnvelopeStatsAccumulator(
        *domains, termination
    )

    with pytest.raises(ValueError, match="positive importance"):
        combined.allocation_importance_for_reference_widths(
            0.9, torch.tensor([2])
        )


def test_universal_termination_envelope_accepts_sparse_positive_end_evidence() -> None:
    importance = torch.tensor([[3.0, 2.0, 1.0]])
    gram = torch.eye(3)
    domains = [
        _FakeDomainAccumulator(importance, gram, routed)
        for routed in (7, 5, 3, 2)
    ]
    termination = _FakeDomainAccumulator(
        torch.tensor([[1.0e-9, 0.0, 0.0]]), gram, 1
    )
    combined = UniversalTerminationCapacityEnvelopeStatsAccumulator(
        *domains, termination
    )

    target = combined.allocation_importance_for_reference_widths(
        0.9, torch.tensor([2])
    )

    assert torch.isfinite(target).all()
    assert bool((target > 0).any().item())


def test_capability_union_uses_minimax_channels_and_trace_balanced_gram() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 1.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0, 2.0])),
        7,
    )
    math = _FakeDomainAccumulator(
        torch.tensor([[1.0, 9.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 6.0, 2.0])),
        5,
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 9.0]]),
        torch.diag(torch.tensor([2.0, 2.0, 8.0])),
        3,
    )

    combined = CapabilityUnionStatsAccumulator(general, math, code)
    normalized = [
        domain._importance / domain._importance.mean()
        for domain in (general, math, code)
    ]
    expected_importance = torch.maximum(
        torch.maximum(normalized[0], normalized[1]), normalized[2]
    ) * general._importance.mean()
    expected_trace = torch.tensor(8.0 * 10.0 * 12.0).pow(1 / 3)

    torch.testing.assert_close(
        combined.camera_importance(0.9), expected_importance
    )
    torch.testing.assert_close(
        combined.allocation_importance(0.9), expected_importance
    )
    assert combined.allocation_reference_importance(0.9) is None
    torch.testing.assert_close(combined.interaction_gram(0).trace(), expected_trace)
    assert torch.linalg.eigvalsh(combined.interaction_gram(0)).min() >= 0


def test_multi_domain_robust_generalizes_minimax_and_skips_zero_trace() -> None:
    domains = (
        _FakeDomainAccumulator(
            torch.tensor([[8.0, 1.0, 1.0, 1.0]]),
            torch.diag(torch.tensor([4.0, 2.0, 2.0, 0.0])),
            7,
        ),
        _FakeDomainAccumulator(
            torch.tensor([[1.0, 8.0, 1.0, 1.0]]),
            torch.diag(torch.tensor([2.0, 6.0, 2.0, 0.0])),
            5,
        ),
        _FakeDomainAccumulator(
            torch.tensor([[1.0, 1.0, 8.0, 1.0]]),
            torch.diag(torch.tensor([2.0, 2.0, 8.0, 0.0])),
            3,
        ),
        _FakeDomainAccumulator(
            torch.tensor([[1.0, 1.0, 1.0, 8.0]]),
            torch.zeros((4, 4)),
            1,
        ),
    )

    combined = MultiDomainRobustStatsAccumulator(*domains)
    normalized = [
        domain._importance / domain._importance.mean() for domain in domains
    ]
    expected_importance = torch.stack(normalized).amax(dim=0)
    expected_importance *= torch.stack(
        [domain._importance.mean() for domain in domains]
    ).prod().pow(1 / 4)
    expected_trace = torch.tensor(8.0 * 10.0 * 12.0).pow(1 / 3)

    torch.testing.assert_close(
        combined.camera_importance(0.9), expected_importance
    )
    torch.testing.assert_close(
        combined.allocation_importance(0.9), expected_importance
    )
    assert combined.allocation_reference_importance(0.9) is None
    torch.testing.assert_close(combined.interaction_gram(0).trace(), expected_trace)
    assert combined.routed_token_counts.tolist() == [16]
    assert combined.reservoir_size_for(0) == 12


def test_phase_capacity_envelope_changes_only_expert_width_evidence() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 2.0, 1.0], [9.0, 2.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0, 2.0])),
        7,
    )
    math = _FakeDomainAccumulator(
        torch.tensor([[2.0, 9.0, 1.0], [2.0, 9.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 6.0, 2.0])),
        5,
    )
    level4 = _FakeDomainAccumulator(
        torch.tensor([[8.0, 2.0, 1.0], [2.0, 8.0, 1.0]]),
        torch.eye(3) * 100,
        3,
    )
    level5 = _FakeDomainAccumulator(
        torch.tensor([[2.0, 8.0, 1.0], [8.0, 2.0, 1.0]]),
        torch.eye(3) * 200,
        3,
    )
    assistant_end = _FakeDomainAccumulator(
        torch.tensor([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0]]),
        torch.eye(3) * 300,
        1,
    )
    combined = PhaseCapacityEnvelopeStatsAccumulator(
        general, math, level4, level5, assistant_end
    )
    base = combined.allocation_importance(0.9)
    target = combined.allocation_importance_for_reference_widths(
        0.9, torch.tensor([2, 2])
    )
    multiplier = target / base

    torch.testing.assert_close(
        combined.camera_importance(0.9),
        combine_domain_importance(
            general._importance,
            math._importance,
            aggregation="domain_anchored",
        ),
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9),
        combined.camera_importance(0.9),
    )
    torch.testing.assert_close(
        multiplier,
        multiplier[:, :1].expand_as(multiplier),
    )
    assert multiplier[1, 0] > multiplier[0, 0]
    assert torch.equal(
        target.argsort(dim=1, descending=True),
        base.argsort(dim=1, descending=True),
    )
    torch.testing.assert_close(
        combined.interaction_gram(0),
        0.9 * general._gram + 0.1 * math._gram,
    )
    assert combined.routed_token_counts.tolist() == [12]
    assert combined.reservoir_size_for(0) == 6


def test_dual_specialist_capacity_preserves_reasoning_channels_and_gram() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[9.0, 2.0, 1.0], [9.0, 2.0, 1.0]]),
        torch.diag(torch.tensor([4.0, 2.0, 2.0])),
        7,
    )
    reasoning = _FakeDomainAccumulator(
        torch.tensor([[2.0, 9.0, 1.0], [2.0, 9.0, 1.0]]),
        torch.diag(torch.tensor([2.0, 6.0, 2.0])),
        5,
    )
    hard_math = _FakeDomainAccumulator(
        torch.tensor([[2.0, 1.0, 1.0], [100.0, 1.0, 1.0]]),
        torch.eye(3) * 100,
        3,
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[1.0, 2.0, 1.0], [1.0, 100.0, 1.0]]),
        torch.eye(3) * 200,
        3,
    )
    combined = DualSpecialistCapacityEnvelopeStatsAccumulator(
        general, reasoning, hard_math, code
    )
    base = combined.allocation_importance(0.9)
    target = combined.allocation_importance_for_reference_widths(
        0.9, torch.tensor([2, 2])
    )
    multiplier = target / base

    torch.testing.assert_close(
        combined.camera_importance(0.9),
        combine_domain_importance(
            general._importance,
            reasoning._importance,
            aggregation="domain_anchored",
        ),
    )
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9),
        combined.camera_importance(0.9),
    )
    torch.testing.assert_close(
        multiplier,
        multiplier[:, :1].expand_as(multiplier),
    )
    assert multiplier[1, 0] > multiplier[0, 0]
    assert torch.equal(
        target.argsort(dim=1, descending=True),
        base.argsort(dim=1, descending=True),
    )
    torch.testing.assert_close(
        combined.interaction_gram(0),
        0.9 * general._gram + 0.1 * reasoning._gram,
    )
    assert combined.routed_token_counts.tolist() == [12]
    assert combined.reservoir_size_for(0) == 6


def test_capability_complement_starts_from_math_geometry_and_only_rewards_code_deficit() -> None:
    general = _FakeDomainAccumulator(
        torch.tensor([[8.0, 2.0, 1.0], [8.0, 1.0, 4.0]]),
        torch.eye(3),
        7,
    )
    math = _FakeDomainAccumulator(
        torch.tensor([[2.0, 8.0, 1.0], [2.0, 1.0, 4.0]]),
        torch.eye(3),
        5,
    )
    code = _FakeDomainAccumulator(
        torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 200.0]]),
        torch.eye(3),
        3,
    )
    combined = CapabilityComplementStatsAccumulator(general, math, code)
    anchored, _, domains = combined._importance_components(0.9)
    reference = torch.sqrt(
        (0.9 * domains[0] + 0.1 * domains[1])
        * torch.maximum(domains[0], domains[1])
    )
    widths = torch.tensor([2, 2])
    target = combined.allocation_importance_for_reference_widths(0.9, widths)
    target_multiplier = target / anchored

    torch.testing.assert_close(combined.allocation_importance(0.9), reference)
    torch.testing.assert_close(
        combined.allocation_reference_importance(0.9), reference
    )
    torch.testing.assert_close(target_multiplier[0], torch.ones(3))
    assert torch.all(target_multiplier[1] > 1)
    torch.testing.assert_close(
        target[1].argsort(descending=True), anchored[1].argsort(descending=True)
    )


class _Expert(nn.Module):
    def __init__(self, hidden_size: int, channel_count: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, channel_count, bias=False)
        self.up_proj = nn.Linear(hidden_size, channel_count, bias=False)
        self.down_proj = nn.Linear(channel_count, hidden_size, bias=False)


class _SparseMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.top_k = 1
        self.norm_topk_prob = True
        self.gate = nn.Linear(2, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(2, 3), _Expert(2, 3)])


class _PackedExperts(nn.Module):
    def __init__(self, legacy: _SparseMoe) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.stack(
                [
                    torch.cat(
                        (expert.gate_proj.weight, expert.up_proj.weight), dim=0
                    )
                    for expert in legacy.experts
                ]
            )
        )
        self.down_proj = nn.Parameter(
            torch.stack(
                [expert.down_proj.weight for expert in legacy.experts]
            )
        )
        self.act_fn = torch.nn.functional.silu


class _PackedRouter(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.top_k = 1

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = hidden_states @ self.weight.T
        probabilities = logits.softmax(dim=-1)
        scores, indices = probabilities.topk(self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        return logits, scores, indices


class _PackedMoe(nn.Module):
    def __init__(self, legacy: _SparseMoe) -> None:
        super().__init__()
        self.gate = _PackedRouter(legacy.gate.weight)
        self.experts = _PackedExperts(legacy)


class _ScaledProbabilityRouter(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.proj.weight.data.copy_(weight)
        self.top_k = 1
        self.register_buffer("per_expert_scale", torch.tensor([2.0, 3.0]))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = self.proj(hidden_states).softmax(dim=-1)
        scores, indices = probabilities.topk(self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        exact_scores = scores * self.per_expert_scale[indices]
        return probabilities, exact_scores, indices


def _module() -> _SparseMoe:
    module = _SparseMoe()
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        module.gate.weight.copy_(torch.eye(2))
        for expert in module.experts:
            for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                projection.weight.copy_(
                    torch.randn(
                        projection.weight.shape, generator=generator
                    )
                )
    return module


def test_layer_accumulator_matches_camera_and_direct_ramp_grams() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=8, reservoir_seed=1234
    )

    accumulator.update(hidden_states[:2])
    accumulator.update(hidden_states[2:])

    gate_weights = torch.stack(
        [expert.gate_proj.weight for expert in module.experts]
    )
    up_weights = torch.stack(
        [expert.up_proj.weight for expert in module.experts]
    )
    down_weights = torch.stack(
        [expert.down_proj.weight for expert in module.experts]
    )
    expected_camera = camera_microexpert_importance(
        hidden_states,
        module.gate.weight,
        gate_weights,
        up_weights,
        down_weights,
        top_k=1,
        alpha=0.9,
        token_chunk_size=2,
    )
    torch.testing.assert_close(accumulator.camera_importance(0.9), expected_camera)

    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        expected_gram = build_interaction_gram(
            intermediate, expert.down_proj.weight
        )
        torch.testing.assert_close(
            accumulator.interaction_gram(expert_index), expected_gram
        )

    assert accumulator.routed_token_counts.tolist() == [2, 2]
    assert accumulator.total_input_tokens == 4


def test_packed_layer_accumulator_matches_legacy_statistics() -> None:
    legacy = _module()
    packed = _PackedMoe(legacy)
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    legacy_stats = Qwen3LayerStatsAccumulator(
        legacy, reservoir_size=8, reservoir_seed=1234
    )
    packed_stats = Qwen3LayerStatsAccumulator(
        packed, reservoir_size=8, reservoir_seed=1234
    )

    legacy_stats.update(hidden_states)
    packed_stats.update(hidden_states)

    torch.testing.assert_close(
        packed_stats.camera_importance(0.9),
        legacy_stats.camera_importance(0.9),
    )
    assert torch.equal(
        packed_stats.routed_token_counts,
        legacy_stats.routed_token_counts,
    )
    for expert_index in range(2):
        torch.testing.assert_close(
            packed_stats.interaction_gram(expert_index),
            legacy_stats.interaction_gram(expert_index),
        )


def test_packed_accumulator_accepts_distinct_router_and_expert_inputs() -> None:
    legacy = _module()
    packed = _PackedMoe(legacy)
    router_hidden = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    expert_hidden = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    accumulator = Qwen3LayerStatsAccumulator(
        packed, reservoir_size=8, reservoir_seed=1234
    )

    accumulator.update(
        router_hidden,
        expert_hidden_states=expert_hidden,
    )

    assert accumulator.total_input_tokens == 2
    assert accumulator.routed_token_counts.tolist() == [1, 1]
    for expert_index in range(2):
        assert accumulator.interaction_gram(expert_index).abs().sum() > 0


def test_gemma_style_router_preserves_per_expert_execution_scale() -> None:
    legacy = _module()
    qwen_block = _PackedMoe(legacy)
    gemma_layer = nn.Module()
    gemma_layer.router = _ScaledProbabilityRouter(legacy.gate.weight)
    gemma_layer.experts = _PackedExperts(legacy)
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    qwen_stats = Qwen3LayerStatsAccumulator(
        qwen_block, reservoir_size=8, reservoir_seed=1234
    )
    gemma_stats = Qwen3LayerStatsAccumulator(
        gemma_layer, reservoir_size=8, reservoir_seed=1234
    )

    qwen_stats.update(hidden_states)
    gemma_stats.update(hidden_states)

    torch.testing.assert_close(
        gemma_stats.camera_importance(0.9),
        qwen_stats.camera_importance(0.9),
    )
    torch.testing.assert_close(
        gemma_stats.interaction_gram(0),
        qwen_stats.interaction_gram(0) * 4,
    )
    torch.testing.assert_close(
        gemma_stats.interaction_gram(1),
        qwen_stats.interaction_gram(1) * 9,
    )


def test_layer_accumulator_exposes_router_weighted_unconditional_phase_mean() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=8, reservoir_seed=1234
    )

    accumulator.update(hidden_states)

    routed_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[routed_experts == expert_index]
        routed = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        expected = routed.sum(dim=0) / hidden_states.shape[0]
        torch.testing.assert_close(
            accumulator.routed_activation_mean(expert_index), expected
        )


def test_layer_accumulator_token_selection_matches_prefiltered_update() -> None:
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    token_selection = torch.tensor([True, False, True, False])
    selected = Qwen3LayerStatsAccumulator(
        _module(), reservoir_size=8, reservoir_seed=1234
    )
    reference = Qwen3LayerStatsAccumulator(
        _module(), reservoir_size=8, reservoir_seed=1234
    )

    selected.update(
        hidden_states, token_selection_mask=token_selection
    )
    reference.update(hidden_states[token_selection])

    torch.testing.assert_close(
        selected.camera_importance(0.9), reference.camera_importance(0.9)
    )
    assert torch.equal(
        selected.routed_token_counts, reference.routed_token_counts
    )
    assert selected.total_input_tokens == reference.total_input_tokens == 2
    for expert_index in range(2):
        torch.testing.assert_close(
            selected.interaction_gram(expert_index),
            reference.interaction_gram(expert_index),
        )


def test_layer_accumulator_rejects_invalid_token_selection_shape() -> None:
    accumulator = Qwen3LayerStatsAccumulator(
        _module(), reservoir_size=8, reservoir_seed=1234
    )

    with pytest.raises(ValueError, match="token_selection_mask"):
        accumulator.update(
            torch.ones(3, 2),
            token_selection_mask=torch.tensor([True, False]),
        )


def test_difference_importance_is_centered_output_energy_with_expert_traffic() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=8, reservoir_seed=1234
    )
    accumulator.update(hidden_states)

    expected = torch.zeros(2, 3)
    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        centered = intermediate - intermediate.mean(dim=0, keepdim=True)
        gram = build_interaction_gram(centered, expert.down_proj.weight)
        traffic = expert_inputs.shape[0] / hidden_states.shape[0]
        expected[expert_index] = gram.diagonal() * traffic

    torch.testing.assert_close(accumulator.difference_importance(), expected)


def test_reservoir_is_bounded_but_counts_all_routed_tokens() -> None:
    module = _module()
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=2, reservoir_seed=7
    )
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [4.0, 0.0]]
    )

    accumulator.update(hidden_states)

    assert accumulator.routed_token_counts.tolist() == [4, 0]
    assert accumulator.reservoir_size_for(0) == 2
    assert accumulator.reservoir_size_for(1) == 0


def test_router_signature_reservoir_stays_aligned_with_activation_reservoir() -> None:
    module = _module()
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=3, reservoir_seed=7
    )
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )

    accumulator.update(hidden_states)

    for expert_index in range(accumulator.expert_count):
        activations, signatures = accumulator.router_neighborhood_samples(
            expert_index
        )
        assert activations.shape[0] == signatures.shape[0]
        assert activations.shape[1] == accumulator.channel_count
        assert signatures.shape[1] == accumulator.expert_count
        if signatures.numel():
            torch.testing.assert_close(
                signatures.sum(dim=1), torch.ones(signatures.shape[0])
            )
            assert (signatures > 0).sum(dim=1).max() <= module.top_k


def test_router_weighted_phase_means_preserve_sample_identity() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [
            [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0]],
            [[-1.0, 2.0], [4.0, 0.0], [1.0, 0.5]],
        ]
    )
    selection = torch.tensor(
        [[True, True, False], [False, True, True]]
    )

    means, routed_counts, selected_counts = router_weighted_phase_means(
        module, hidden_states, token_selection_mask=selection
    )

    assert selected_counts.tolist() == [2, 2]
    for sample_index in range(2):
        reference = Qwen3LayerStatsAccumulator(
            module, reservoir_size=8, reservoir_seed=1234
        )
        reference.update(
            hidden_states[sample_index],
            token_selection_mask=selection[sample_index],
        )
        torch.testing.assert_close(
            means[sample_index],
            torch.stack(
                [
                    reference.routed_activation_mean(expert_index)
                    for expert_index in range(reference.expert_count)
                ]
            ),
        )
        torch.testing.assert_close(
            routed_counts[sample_index], reference.routed_token_counts
        )


def test_local_separation_gram_reports_pairs_and_matches_ordinary_trace() -> None:
    module = _module()
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=8, reservoir_seed=1234
    )
    accumulator.update(
        torch.tensor([[3.0, 0.0], [2.0, -1.0], [1.0, 0.5], [4.0, 0.0]])
    )

    local, trace = accumulator.local_separation_gram(0, return_trace=True)

    assert trace["pair_count"] >= 2
    assert trace["fallback"] is False
    torch.testing.assert_close(local.trace(), accumulator.interaction_gram(0).trace())


def test_unrouted_expert_has_zero_interaction_gram() -> None:
    module = _module()
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=2, reservoir_seed=7
    )
    accumulator.update(torch.tensor([[3.0, 0.0], [2.0, 0.0]]))

    gram = accumulator.interaction_gram(1)

    torch.testing.assert_close(gram, torch.zeros(3, 3))


def test_next_rmsnorm_scale_weights_camera_energy_and_interaction_gram() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    output_scale = torch.tensor([2.0, 0.5])
    accumulator = Qwen3LayerStatsAccumulator(
        module,
        reservoir_size=8,
        reservoir_seed=1234,
        output_metric_scale=output_scale,
    )

    accumulator.update(hidden_states)

    gate_weights = torch.stack(
        [expert.gate_proj.weight for expert in module.experts]
    )
    up_weights = torch.stack(
        [expert.up_proj.weight for expert in module.experts]
    )
    weighted_down = torch.stack(
        [
            expert.down_proj.weight * output_scale[:, None]
            for expert in module.experts
        ]
    )
    expected_camera = camera_microexpert_importance(
        hidden_states,
        module.gate.weight,
        gate_weights,
        up_weights,
        weighted_down,
        top_k=1,
        alpha=0.9,
        token_chunk_size=4,
    )
    torch.testing.assert_close(accumulator.camera_importance(0.9), expected_camera)

    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        expected_gram = build_interaction_gram(
            intermediate,
            expert.down_proj.weight * output_scale[:, None],
        )
        torch.testing.assert_close(
            accumulator.interaction_gram(expert_index), expected_gram
        )


def test_token_metric_scale_weights_camera_energy_and_interaction_gram() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    token_scale = torch.tensor([1.0, 4.0, 0.25, 2.25])
    accumulator = Qwen3LayerStatsAccumulator(
        module, reservoir_size=8, reservoir_seed=1234
    )

    accumulator.update(hidden_states, token_metric_scale=token_scale)

    router_experts = (hidden_states @ module.gate.weight.T).argmax(dim=-1)
    expected_camera = torch.zeros(2, 3)
    for expert_index, expert in enumerate(module.experts):
        selected = router_experts == expert_index
        expert_inputs = hidden_states[selected]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        routed = intermediate
        weighted_square = routed.square() * token_scale[selected, None]
        activation_energy = 0.1 * weighted_square.sum(dim=0)
        activation_energy += 0.9 * weighted_square.amax(dim=0)
        expected_camera[expert_index] = (
            activation_energy * expert.down_proj.weight.square().sum(dim=0)
        )
        expected_gram = build_interaction_gram(
            routed * token_scale[selected, None].sqrt(),
            expert.down_proj.weight,
        )
        torch.testing.assert_close(
            accumulator.interaction_gram(expert_index), expected_gram
        )

    torch.testing.assert_close(accumulator.camera_importance(0.9), expected_camera)


def test_token_metric_scale_rejects_invalid_values() -> None:
    accumulator = Qwen3LayerStatsAccumulator(
        _module(), reservoir_size=8, reservoir_seed=1234
    )
    hidden_states = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    for invalid in (
        torch.ones(3),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("inf")]),
    ):
        try:
            accumulator.update(hidden_states, token_metric_scale=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid token metric scale was accepted")


def test_accumulator_can_select_centered_difference_interaction_metric() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module,
        reservoir_size=8,
        reservoir_seed=1234,
        interaction_metric="centered_difference",
        difference_regularization=0.01,
    )
    accumulator.update(hidden_states)

    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        expected = build_centered_difference_interaction_gram(
            intermediate,
            expert.down_proj.weight,
            regularization=0.01,
        )
        torch.testing.assert_close(accumulator.interaction_gram(expert_index), expected)
        torch.testing.assert_close(
            accumulator.reconstruction_gram(expert_index),
            build_interaction_gram(intermediate, expert.down_proj.weight),
        )


def test_accumulator_rejects_invalid_interaction_metric() -> None:
    with pytest.raises(ValueError, match="interaction_metric"):
        Qwen3LayerStatsAccumulator(
            _module(), interaction_metric="unknown"  # type: ignore[arg-type]
        )


def test_accumulator_can_select_energy_calibrated_difference_metric() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module,
        reservoir_size=8,
        reservoir_seed=1234,
        interaction_metric="energy_calibrated_difference",
        difference_strength=1.0,
    )
    accumulator.update(hidden_states)

    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        ordinary = build_interaction_gram(intermediate, expert.down_proj.weight)
        centered = intermediate - intermediate.mean(dim=0, keepdim=True)
        expected = ordinary + build_interaction_gram(
            centered, expert.down_proj.weight
        )
        torch.testing.assert_close(accumulator.interaction_gram(expert_index), expected)


def test_accumulator_can_apply_difference_only_to_coefficient_recovery() -> None:
    module = _module()
    hidden_states = torch.tensor(
        [[3.0, 0.0], [2.0, -1.0], [0.0, 4.0], [-1.0, 2.0]]
    )
    accumulator = Qwen3LayerStatsAccumulator(
        module,
        reservoir_size=8,
        reservoir_seed=1234,
        coefficient_difference_strength=1.0,
    )
    accumulator.update(hidden_states)

    router_experts = hidden_states.argmax(dim=1)
    for expert_index, expert in enumerate(module.experts):
        expert_inputs = hidden_states[router_experts == expert_index]
        intermediate = torch.nn.functional.silu(
            expert.gate_proj(expert_inputs)
        ) * expert.up_proj(expert_inputs)
        ordinary = build_interaction_gram(intermediate, expert.down_proj.weight)
        expected_recovery = build_energy_calibrated_difference_interaction_gram(
            intermediate,
            expert.down_proj.weight,
            strength=1.0,
        )
        torch.testing.assert_close(accumulator.interaction_gram(expert_index), ordinary)
        torch.testing.assert_close(
            accumulator.reconstruction_gram(expert_index), expected_recovery
        )


def test_accumulator_rejects_negative_coefficient_difference_strength() -> None:
    with pytest.raises(ValueError, match="coefficient_difference_strength"):
        Qwen3LayerStatsAccumulator(
            _module(), coefficient_difference_strength=-1.0
        )
