"""Streaming Qwen3-MoE statistics for CAMERA and RAMP selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .moe_adapter import resolve_moe_layer

from .core import (
    build_centered_difference_interaction_gram,
    build_energy_calibrated_difference_interaction_gram,
    build_interaction_gram,
    build_router_neighborhood_local_separation_gram,
)


def combine_domain_importance(
    general: torch.Tensor,
    reasoning: torch.Tensor,
    *,
    aggregation: str = "domain_robust",
) -> torch.Tensor:
    """Combine per-domain importance without letting corpus scale dominate."""

    if aggregation not in {
        "domain_robust",
        "domain_anchored",
        "bounded_domain_anchored",
        "anchored_capacity_trust_region",
        "anchored_robust_capacity",
        "anchored_geometric_capacity",
        "anchored_difference_capacity",
        "hierarchical_robust",
        "bounded_hierarchical_robust",
    }:
        raise ValueError("unsupported domain importance aggregation")
    general = torch.as_tensor(general, dtype=torch.float32)
    reasoning = torch.as_tensor(
        reasoning, dtype=torch.float32, device=general.device
    )
    if general.shape != reasoning.shape or general.ndim != 2:
        raise ValueError("domain importance tensors must share [expert, channel] shape")
    if (
        not torch.isfinite(general).all()
        or not torch.isfinite(reasoning).all()
        or (general < 0).any()
        or (reasoning < 0).any()
    ):
        raise ValueError("domain importance must be finite and non-negative")
    general_scale = general.mean()
    reasoning_scale = reasoning.mean()
    epsilon = torch.finfo(general.dtype).eps
    if general_scale <= epsilon:
        return reasoning.clone()
    if reasoning_scale <= epsilon:
        return general.clone()
    reference_scale = torch.sqrt(general_scale * reasoning_scale)
    normalized_general = general / general_scale
    normalized_reasoning = reasoning / reasoning_scale
    if aggregation in {
        "domain_robust",
        "hierarchical_robust",
        "bounded_hierarchical_robust",
    }:
        normalized = torch.maximum(normalized_general, normalized_reasoning)
    else:
        normalized = 0.9 * normalized_general + 0.1 * normalized_reasoning
    return normalized * reference_scale


class DomainRobustStatsAccumulator:
    """Expose balanced general/reasoning statistics to the existing selector."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        *,
        aggregation: str = "domain_robust",
    ) -> None:
        attributes = ("expert_count", "channel_count", "device")
        if any(getattr(general, name) != getattr(reasoning, name) for name in attributes):
            raise ValueError("domain accumulators must have identical dimensions and device")
        self.general = general
        self.reasoning = reasoning
        self.expert_count = general.expert_count
        self.channel_count = general.channel_count
        self.device = general.device
        if aggregation not in {
            "domain_robust",
            "domain_anchored",
            "bounded_domain_anchored",
            "anchored_capacity_trust_region",
            "anchored_robust_capacity",
            "anchored_geometric_capacity",
            "anchored_difference_capacity",
            "hierarchical_robust",
            "bounded_hierarchical_robust",
        }:
            raise ValueError("unsupported domain statistics aggregation")
        self.aggregation = aggregation
        self.general_weight = (
            0.9
            if aggregation
            in {
                "domain_anchored",
                "bounded_domain_anchored",
                "anchored_capacity_trust_region",
                "anchored_robust_capacity",
                "anchored_geometric_capacity",
                "anchored_difference_capacity",
            }
            else 0.5
        )
        self.reasoning_weight = 1.0 - self.general_weight

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return self.general.routed_token_counts + self.reasoning.routed_token_counts

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        aggregation = (
            "domain_anchored"
            if self.aggregation in {
                "anchored_robust_capacity",
                "anchored_geometric_capacity",
                "anchored_difference_capacity",
            }
            else self.aggregation
        )
        return combine_domain_importance(
            self.general.camera_importance(alpha),
            self.reasoning.camera_importance(alpha),
            aggregation=aggregation,
        )

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        """Return the signal used only for model-wide expert width budgets."""

        if self.aggregation in {
            "anchored_capacity_trust_region",
            "hierarchical_robust",
        }:
            return self.general.camera_importance(alpha)
        if self.aggregation == "anchored_robust_capacity":
            return combine_domain_importance(
                self.general.camera_importance(alpha),
                self.reasoning.camera_importance(alpha),
                aggregation="domain_robust",
            )
        if self.aggregation == "anchored_geometric_capacity":
            anchored = self.camera_importance(alpha)
            robust = combine_domain_importance(
                self.general.camera_importance(alpha),
                self.reasoning.camera_importance(alpha),
                aggregation="domain_robust",
            )
            return torch.sqrt(anchored * robust)
        if self.aggregation == "anchored_difference_capacity":
            robust = combine_domain_importance(
                self.general.camera_importance(alpha),
                self.reasoning.camera_importance(alpha),
                aggregation="domain_robust",
            )
            robust_difference = combine_domain_importance(
                self.general.difference_importance(),
                self.reasoning.difference_importance(),
                aggregation="domain_robust",
            )
            return torch.sqrt(robust * robust_difference)
        return self.camera_importance(alpha)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor | None:
        """Return a general-domain width anchor for bounded migration."""

        if self.aggregation in {
            "bounded_domain_anchored",
            "bounded_hierarchical_robust",
        }:
            return self.general.camera_importance(alpha)
        if self.aggregation in {
            "anchored_capacity_trust_region",
            "anchored_robust_capacity",
            "anchored_geometric_capacity",
            "anchored_difference_capacity",
        }:
            return self.camera_importance(alpha)
        return None

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            self.general_weight * self.general.interaction_gram(expert_index)
            + self.reasoning_weight * self.reasoning.interaction_gram(expert_index)
        )

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            self.general_weight * self.general.reconstruction_gram(expert_index)
            + self.reasoning_weight
            * self.reasoning.reconstruction_gram(expert_index)
        )

    def reservoir_size_for(self, expert_index: int) -> int:
        return self.general.reservoir_size_for(
            expert_index
        ) + self.reasoning.reservoir_size_for(expert_index)


class CapabilityEnvelopeStatsAccumulator:
    """Protect independent math and code capacity without spiky channel ranking."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
    ) -> None:
        attributes = ("expert_count", "channel_count", "device")
        domains = (general, math, code)
        if any(
            getattr(domain, name) != getattr(general, name)
            for domain in domains[1:]
            for name in attributes
        ):
            raise ValueError("capability accumulators must share dimensions and device")
        self.general = general
        self.math = math
        self.code = code
        self.expert_count = general.expert_count
        self.channel_count = general.channel_count
        self.device = general.device

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return (
            self.general.routed_token_counts
            + self.math.routed_token_counts
            + self.code.routed_token_counts
        )

    def _importance_components(
        self, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        values = [
            domain.camera_importance(alpha).to(dtype=torch.float32)
            for domain in (self.general, self.math, self.code)
        ]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("capability importance tensors must share shape")
        if any(
            not torch.isfinite(value).all() or (value < 0).any()
            for value in values
        ):
            raise ValueError("capability importance must be finite and non-negative")
        scales = torch.stack([value.mean() for value in values])
        if (scales <= 0).any():
            raise ValueError("every capability domain must contain positive importance")
        normalized = [value / scale for value, scale in zip(values, scales)]
        reference_scale = scales.prod().pow(1 / len(values))
        scaled = tuple(value * reference_scale for value in normalized)
        anchored = (
            0.9 * scaled[0]
            + 0.05 * scaled[1]
            + 0.05 * scaled[2]
        )
        envelope = torch.maximum(
            torch.maximum(scaled[0], scaled[1]), scaled[2]
        )
        return anchored, envelope, scaled

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        anchored, _, _ = self._importance_components(alpha)
        return anchored

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        anchored, _, _ = self._importance_components(alpha)
        return anchored

    def allocation_importance_for_reference_widths(
        self, alpha: float, reference_widths: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast expert-level capability evidence over anchored channels."""

        anchored, _, domains = self._importance_components(alpha)
        widths = torch.as_tensor(reference_widths, device=self.device)
        if widths.shape != (self.expert_count,):
            raise ValueError("reference widths must contain one value per expert")
        if widths.dtype == torch.bool or widths.is_floating_point():
            raise ValueError("reference widths must contain integers")
        widths = widths.to(torch.long)
        if (widths < 0).any() or (widths > self.channel_count).any():
            raise ValueError("reference widths must lie inside the channel range")
        evidence_widths = widths.clamp_min(1)
        order = torch.argsort(anchored, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            self.channel_count, device=self.device
        ).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        evidence_mask = ranks < evidence_widths.unsqueeze(1)
        anchored_mass = (anchored * evidence_mask).sum(dim=1)
        domain_mass = torch.stack(
            [(domain * evidence_mask).sum(dim=1) for domain in domains]
        )
        robust_mass = domain_mass.amax(dim=0)
        epsilon = torch.finfo(anchored.dtype).eps
        multiplier = torch.sqrt(
            robust_mass / anchored_mass.clamp_min(epsilon)
        )
        return anchored * multiplier.unsqueeze(1)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor:
        return self.camera_importance(alpha)

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            0.9 * self.general.interaction_gram(expert_index)
            + 0.05 * self.math.interaction_gram(expert_index)
            + 0.05 * self.code.interaction_gram(expert_index)
        )

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            0.9 * self.general.reconstruction_gram(expert_index)
            + 0.05 * self.math.reconstruction_gram(expert_index)
            + 0.05 * self.code.reconstruction_gram(expert_index)
        )

    def reservoir_size_for(self, expert_index: int) -> int:
        return sum(
            domain.reservoir_size_for(expert_index)
            for domain in (self.general, self.math, self.code)
        )


class UniversalCapabilityEnvelopeStatsAccumulator:
    """Balance reasoning, hard-math, and code under one general anchor."""

    _DEFAULT_GENERAL_WEIGHT = 0.9

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        hard_math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
        general_weight: float = _DEFAULT_GENERAL_WEIGHT,
    ) -> None:
        domains = (general, reasoning, hard_math, code)
        attributes = ("expert_count", "channel_count", "device")
        if any(
            getattr(domain, name) != getattr(general, name)
            for domain in domains[1:]
            for name in attributes
        ):
            raise ValueError(
                "universal capability accumulators must share dimensions and device"
            )
        self.domains = domains
        self.general = general
        self.reasoning = reasoning
        self.hard_math = hard_math
        self.code = code
        self.expert_count = general.expert_count
        self.channel_count = general.channel_count
        self.device = general.device
        if not 0 <= general_weight <= 1:
            raise ValueError("general anchor weight must be in [0, 1]")
        self.general_weight = float(general_weight)

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return torch.stack(
            [domain.routed_token_counts for domain in self.domains]
        ).sum(dim=0)

    def _importance_components(
        self, alpha: float
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        values = tuple(
            domain.camera_importance(alpha).to(dtype=torch.float32)
            for domain in self.domains
        )
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError(
                "universal capability importance tensors must share shape"
            )
        if any(
            not torch.isfinite(value).all() or (value < 0).any()
            for value in values
        ):
            raise ValueError(
                "universal capability importance must be finite and non-negative"
            )
        scales = torch.stack([value.mean() for value in values])
        if (scales <= 0).any():
            raise ValueError(
                "every universal capability domain must contain positive importance"
            )
        reference_scale = scales.prod().pow(1 / len(values))
        scaled = tuple(
            value / scale * reference_scale
            for value, scale in zip(values, scales)
        )
        specialist_weight = (1 - self.general_weight) / (len(scaled) - 1)
        anchored = self.general_weight * scaled[0] + specialist_weight * sum(
            scaled[1:]
        )
        return anchored, scaled

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        anchored, _ = self._importance_components(alpha)
        return anchored

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        return self.camera_importance(alpha)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor:
        return self.camera_importance(alpha)

    def allocation_importance_for_reference_widths(
        self, alpha: float, reference_widths: torch.Tensor
    ) -> torch.Tensor:
        """Use a domain-max expert multiplier without changing channel order."""

        anchored, domains = self._importance_components(alpha)
        multiplier = self.capacity_multipliers_for_reference_widths(
            alpha,
            reference_widths,
            _components=(anchored, domains),
        )
        return anchored * multiplier.unsqueeze(1)

    def capacity_multipliers_for_reference_widths(
        self,
        alpha: float,
        reference_widths: torch.Tensor,
        *,
        _components: tuple[
            torch.Tensor, tuple[torch.Tensor, ...]
        ] | None = None,
    ) -> torch.Tensor:
        """Return the expert-wise capability-envelope multiplier ``rho_e``."""

        anchored, domains = (
            self._importance_components(alpha)
            if _components is None
            else _components
        )
        widths = torch.as_tensor(reference_widths, device=self.device)
        if widths.shape != (self.expert_count,):
            raise ValueError("reference widths must contain one value per expert")
        if widths.dtype == torch.bool or widths.is_floating_point():
            raise ValueError("reference widths must contain integers")
        widths = widths.to(torch.long)
        if (widths < 0).any() or (widths > self.channel_count).any():
            raise ValueError("reference widths must lie inside the channel range")
        evidence_widths = widths.clamp_min(1)
        order = torch.argsort(anchored, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(self.channel_count, device=self.device).expand_as(order),
        )
        evidence_mask = ranks < evidence_widths.unsqueeze(1)
        anchored_mass = (anchored * evidence_mask).sum(dim=1)
        robust_mass = torch.stack(
            [(domain * evidence_mask).sum(dim=1) for domain in domains]
        ).amax(dim=0)
        epsilon = torch.finfo(anchored.dtype).eps
        multiplier = torch.sqrt(
            robust_mass / anchored_mass.clamp_min(epsilon)
        )
        if not torch.isfinite(multiplier).all() or (multiplier < 0).any():
            raise RuntimeError("capability multipliers must be finite and non-negative")
        return multiplier

    def _anchored_gram(
        self, expert_index: int, *, reconstruction: bool
    ) -> torch.Tensor:
        method = "reconstruction_gram" if reconstruction else "interaction_gram"
        grams = [
            getattr(domain, method)(expert_index) for domain in self.domains
        ]
        specialist_weight = (1 - self.general_weight) / (len(grams) - 1)
        return self.general_weight * grams[0] + specialist_weight * sum(grams[1:])

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return self._anchored_gram(expert_index, reconstruction=False)

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return self._anchored_gram(expert_index, reconstruction=True)

    def reservoir_size_for(self, expert_index: int) -> int:
        return sum(
            domain.reservoir_size_for(expert_index) for domain in self.domains
        )


class UniversalEqualDomainStatsAccumulator:
    """Expose four mean-normalized domains as an equal-weight CAMERA signal."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        hard_math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
    ) -> None:
        domains = (general, reasoning, hard_math, code)
        attributes = ("expert_count", "channel_count", "device")
        if any(
            getattr(domain, name) != getattr(general, name)
            for domain in domains[1:]
            for name in attributes
        ):
            raise ValueError(
                "equal-domain accumulators must share dimensions and device"
            )
        self.domains = domains
        self.expert_count = general.expert_count
        self.channel_count = general.channel_count
        self.device = general.device

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return torch.stack(
            [domain.routed_token_counts for domain in self.domains]
        ).sum(dim=0)

    def _normalized_importance(self, alpha: float) -> torch.Tensor:
        values = tuple(
            domain.camera_importance(alpha).to(dtype=torch.float32)
            for domain in self.domains
        )
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("equal-domain importance tensors must share shape")
        if any(
            not torch.isfinite(value).all() or (value < 0).any()
            for value in values
        ):
            raise ValueError(
                "equal-domain importance must be finite and non-negative"
            )
        scales = torch.stack([value.mean() for value in values])
        if (scales <= 0).any():
            raise ValueError(
                "every equal-domain input must contain positive importance"
            )
        reference_scale = scales.prod().pow(1 / len(values))
        normalized = tuple(
            value / scale * reference_scale
            for value, scale in zip(values, scales)
        )
        return sum(normalized) / len(normalized)

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        return self._normalized_importance(alpha)

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        return self.camera_importance(alpha)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor | None:
        return None

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return sum(
            domain.interaction_gram(expert_index) for domain in self.domains
        ) / len(self.domains)

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return sum(
            domain.reconstruction_gram(expert_index) for domain in self.domains
        ) / len(self.domains)

    def reservoir_size_for(self, expert_index: int) -> int:
        return sum(
            domain.reservoir_size_for(expert_index) for domain in self.domains
        )


class UniversalTerminationCapacityEnvelopeStatsAccumulator(
    UniversalCapabilityEnvelopeStatsAccumulator
):
    """Add assistant termination evidence to expert capacity only."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        hard_math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
        termination: Qwen3LayerStatsAccumulator,
    ) -> None:
        super().__init__(general, reasoning, hard_math, code)
        attributes = ("expert_count", "channel_count", "device")
        if any(
            getattr(termination, name) != getattr(general, name)
            for name in attributes
        ):
            raise ValueError(
                "termination accumulator must share dimensions and device"
            )
        self.termination = termination

    @property
    def termination_routed_token_counts(self) -> torch.Tensor:
        return self.termination.routed_token_counts

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return super().routed_token_counts + self.termination.routed_token_counts

    def allocation_importance_for_reference_widths(
        self, alpha: float, reference_widths: torch.Tensor
    ) -> torch.Tensor:
        """Expand robust mass with end evidence while preserving anchor order."""

        anchored, domains = self._importance_components(alpha)
        termination = self.termination.camera_importance(alpha).to(
            dtype=torch.float32
        )
        if termination.shape != anchored.shape:
            raise ValueError(
                "termination importance must share the anchor shape"
            )
        if not torch.isfinite(termination).all() or (termination < 0).any():
            raise ValueError(
                "termination importance must be finite and non-negative"
            )
        termination_scale = termination.mean()
        if termination_scale <= 0:
            raise ValueError("termination domain must contain positive importance")
        termination = termination / termination_scale * domains[0].mean()

        widths = torch.as_tensor(reference_widths, device=self.device)
        if widths.shape != (self.expert_count,):
            raise ValueError("reference widths must contain one value per expert")
        if widths.dtype == torch.bool or widths.is_floating_point():
            raise ValueError("reference widths must contain integers")
        widths = widths.to(torch.long)
        if (widths < 0).any() or (widths > self.channel_count).any():
            raise ValueError("reference widths must lie inside the channel range")
        evidence_widths = widths.clamp_min(1)
        order = torch.argsort(anchored, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(self.channel_count, device=self.device).expand_as(order),
        )
        evidence_mask = ranks < evidence_widths.unsqueeze(1)
        anchored_mass = (anchored * evidence_mask).sum(dim=1)
        robust_mass = torch.stack(
            [
                *((domain * evidence_mask).sum(dim=1) for domain in domains),
                (termination * evidence_mask).sum(dim=1),
            ]
        ).amax(dim=0)
        multiplier = torch.sqrt(
            robust_mass
            / anchored_mass.clamp_min(torch.finfo(anchored.dtype).eps)
        )
        return anchored * multiplier.unsqueeze(1)

    def reservoir_size_for(self, expert_index: int) -> int:
        return super().reservoir_size_for(
            expert_index
        ) + self.termination.reservoir_size_for(expert_index)


class _SpecialistCapacityEnvelopeStatsAccumulator:
    """Move only reasoning-anchored expert widths for specialist deficits."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        *specialists: Qwen3LayerStatsAccumulator,
    ) -> None:
        if not specialists:
            raise ValueError("at least one specialist accumulator is required")
        domains = (general, reasoning, *specialists)
        attributes = ("expert_count", "channel_count", "device")
        if any(
            getattr(domain, name) != getattr(general, name)
            for domain in domains[1:]
            for name in attributes
        ):
            raise ValueError(
                "specialist capacity accumulators must share dimensions and device"
            )
        self.general = general
        self.math = reasoning
        self.phase_domains = specialists
        self.expert_count = general.expert_count
        self.channel_count = general.channel_count
        self.device = general.device

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return self.general.routed_token_counts + self.math.routed_token_counts

    def _importance_components(
        self, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        domains = (self.general, self.math, *self.phase_domains)
        values = [
            domain.camera_importance(alpha).to(dtype=torch.float32)
            for domain in domains
        ]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("specialist capacity importance tensors must share shape")
        if any(
            not torch.isfinite(value).all() or (value < 0).any()
            for value in values
        ):
            raise ValueError(
                "specialist capacity importance must be finite and non-negative"
            )
        scales = torch.stack([value.mean() for value in values])
        if (scales <= 0).any():
            raise ValueError(
                "every specialist capacity domain must contain positive importance"
            )
        reference_scale = torch.sqrt(scales[0] * scales[1])
        scaled = tuple(
            value / scale * reference_scale
            for value, scale in zip(values, scales)
        )
        anchored = 0.9 * scaled[0] + 0.1 * scaled[1]
        robust_base = torch.maximum(scaled[0], scaled[1])
        geometric_base = torch.sqrt(anchored * robust_base)
        return anchored, robust_base, geometric_base, scaled[2:]

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        anchored, _, _, _ = self._importance_components(alpha)
        return anchored

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        _, _, geometric_base, _ = self._importance_components(alpha)
        return geometric_base

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor:
        return self.camera_importance(alpha)

    def allocation_importance_for_reference_widths(
        self, alpha: float, reference_widths: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast specialist deficits per expert without changing channel order."""

        anchored, robust_base, geometric_base, phase_domains = (
            self._importance_components(alpha)
        )
        widths = torch.as_tensor(reference_widths, device=self.device)
        if widths.shape != (self.expert_count,):
            raise ValueError("reference widths must contain one value per expert")
        if widths.dtype == torch.bool or widths.is_floating_point():
            raise ValueError("reference widths must contain integers")
        widths = widths.to(torch.long)
        if (widths < 0).any() or (widths > self.channel_count).any():
            raise ValueError("reference widths must lie inside the channel range")
        evidence_widths = widths.clamp_min(1)
        order = torch.argsort(anchored, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(self.channel_count, device=self.device).expand_as(order),
        )
        evidence_mask = ranks < evidence_widths.unsqueeze(1)
        base_mass = (robust_base * evidence_mask).sum(dim=1)
        phase_mass = torch.stack(
            [(domain * evidence_mask).sum(dim=1) for domain in phase_domains]
        ).amax(dim=0)
        robust_mass = torch.maximum(base_mass, phase_mass)
        epsilon = torch.finfo(anchored.dtype).eps
        multiplier = torch.sqrt(
            robust_mass / base_mass.clamp_min(epsilon)
        )
        return geometric_base * multiplier.unsqueeze(1)

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            0.9 * self.general.interaction_gram(expert_index)
            + 0.1 * self.math.interaction_gram(expert_index)
        )

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return (
            0.9 * self.general.reconstruction_gram(expert_index)
            + 0.1 * self.math.reconstruction_gram(expert_index)
        )

    def reservoir_size_for(self, expert_index: int) -> int:
        return self.general.reservoir_size_for(
            expert_index
        ) + self.math.reservoir_size_for(expert_index)


class PhaseCapacityEnvelopeStatsAccumulator(
    _SpecialistCapacityEnvelopeStatsAccumulator
):
    """Move only anchored expert widths for hard-level and stopping deficits."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        math: Qwen3LayerStatsAccumulator,
        level4: Qwen3LayerStatsAccumulator,
        level5: Qwen3LayerStatsAccumulator,
        assistant_end: Qwen3LayerStatsAccumulator,
    ) -> None:
        super().__init__(general, math, level4, level5, assistant_end)


class DualSpecialistCapacityEnvelopeStatsAccumulator(
    _SpecialistCapacityEnvelopeStatsAccumulator
):
    """Keep reasoning channels fixed while math and code move expert widths."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        reasoning: Qwen3LayerStatsAccumulator,
        hard_math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
    ) -> None:
        super().__init__(general, reasoning, hard_math, code)


class MultiDomainRobustStatsAccumulator:
    """Expose a scale- and trace-balanced minimax union over named domains."""

    def __init__(self, *domains: Qwen3LayerStatsAccumulator) -> None:
        if len(domains) < 2:
            raise ValueError("multi-domain statistics require at least two domains")
        reference = domains[0]
        attributes = ("expert_count", "channel_count", "device")
        if any(
            getattr(domain, name) != getattr(reference, name)
            for domain in domains[1:]
            for name in attributes
        ):
            raise ValueError("multi-domain accumulators must share dimensions and device")
        self.domains = domains
        self.expert_count = reference.expert_count
        self.channel_count = reference.channel_count
        self.device = reference.device

    @property
    def routed_token_counts(self) -> torch.Tensor:
        return torch.stack(
            [domain.routed_token_counts for domain in self.domains]
        ).sum(dim=0)

    def _scale_balanced_importance(self, alpha: float) -> torch.Tensor:
        values = [
            domain.camera_importance(alpha).to(dtype=torch.float32)
            for domain in self.domains
        ]
        reference = values[0]
        if any(value.shape != reference.shape for value in values[1:]):
            raise ValueError("multi-domain importance tensors must share shape")
        if any(
            not torch.isfinite(value).all() or (value < 0).any()
            for value in values
        ):
            raise ValueError("multi-domain importance must be finite and non-negative")
        scales = torch.stack([value.mean() for value in values])
        if (scales <= 0).any():
            raise ValueError("every robust domain must contain positive importance")
        reference_scale = scales.prod().pow(1 / len(values))
        normalized = torch.stack(
            [value / scale for value, scale in zip(values, scales)]
        )
        return normalized.amax(dim=0) * reference_scale

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        return self._scale_balanced_importance(alpha)

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        return self.camera_importance(alpha)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> None:
        del alpha
        return None

    def _trace_balanced_gram(
        self, expert_index: int, *, reconstruction: bool
    ) -> torch.Tensor:
        method = (
            "reconstruction_gram" if reconstruction else "interaction_gram"
        )
        grams = [
            getattr(domain, method)(expert_index)
            for domain in self.domains
        ]
        traces = torch.stack([gram.trace().clamp_min(0) for gram in grams])
        epsilon = torch.finfo(grams[0].dtype).eps
        positive = traces > epsilon
        if not positive.any():
            return torch.zeros_like(grams[0])
        reference_trace = traces[positive].log().mean().exp()
        balanced = [
            gram * (reference_trace / trace)
            for gram, trace, keep in zip(grams, traces, positive)
            if bool(keep.item())
        ]
        return torch.stack(balanced).mean(dim=0)

    def interaction_gram(self, expert_index: int) -> torch.Tensor:
        return self._trace_balanced_gram(expert_index, reconstruction=False)

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return self._trace_balanced_gram(expert_index, reconstruction=True)

    def reservoir_size_for(self, expert_index: int) -> int:
        return sum(
            domain.reservoir_size_for(expert_index) for domain in self.domains
        )


class CapabilityUnionStatsAccumulator(MultiDomainRobustStatsAccumulator):
    """Expose a scale-balanced minimax union across general, math, and code."""

    def __init__(
        self,
        general: Qwen3LayerStatsAccumulator,
        math: Qwen3LayerStatsAccumulator,
        code: Qwen3LayerStatsAccumulator,
    ) -> None:
        super().__init__(general, math, code)
        self.general = general
        self.math = math
        self.code = code


class CapabilityComplementStatsAccumulator(CapabilityEnvelopeStatsAccumulator):
    """Start from general/math geometry and spend capacity only on code deficits."""

    @staticmethod
    def _general_math_geometric(
        domains: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        general, math, _ = domains
        anchored = 0.9 * general + 0.1 * math
        robust = torch.maximum(general, math)
        return torch.sqrt(anchored * robust)

    def allocation_importance(self, alpha: float = 0.9) -> torch.Tensor:
        _, _, domains = self._importance_components(alpha)
        return self._general_math_geometric(domains)

    def allocation_reference_importance(
        self, alpha: float = 0.9
    ) -> torch.Tensor:
        return self.allocation_importance(alpha)

    def allocation_importance_for_reference_widths(
        self, alpha: float, reference_widths: torch.Tensor
    ) -> torch.Tensor:
        """Reward only code evidence missing from the general/math envelope."""

        anchored, _, domains = self._importance_components(alpha)
        widths = torch.as_tensor(reference_widths, device=self.device)
        if widths.shape != (self.expert_count,):
            raise ValueError("reference widths must contain one value per expert")
        if widths.dtype == torch.bool or widths.is_floating_point():
            raise ValueError("reference widths must contain integers")
        widths = widths.to(torch.long)
        if (widths < 0).any() or (widths > self.channel_count).any():
            raise ValueError("reference widths must lie inside the channel range")
        evidence_widths = widths.clamp_min(1)
        order = torch.argsort(anchored, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(self.channel_count, device=self.device).expand_as(order),
        )
        evidence_mask = ranks < evidence_widths.unsqueeze(1)
        general_mass = (domains[0] * evidence_mask).sum(dim=1)
        math_mass = (domains[1] * evidence_mask).sum(dim=1)
        code_mass = (domains[2] * evidence_mask).sum(dim=1)
        base_mass = torch.maximum(general_mass, math_mass)
        robust_mass = torch.maximum(base_mass, code_mass)
        epsilon = torch.finfo(anchored.dtype).eps
        multiplier = torch.sqrt(robust_mass / base_mass.clamp_min(epsilon))
        return anchored * multiplier.unsqueeze(1)


@torch.no_grad()
def router_weighted_phase_means(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    token_selection_mask: torch.Tensor,
    token_metric_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample phase means with unrouted expert tokens as zero."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [samples, tokens, hidden]")
    if not hasattr(module, "gate") or not hasattr(module, "experts"):
        raise ValueError("module must expose Qwen3-MoE gate and experts")
    sample_count, token_count, hidden_size = hidden_states.shape
    selection = torch.as_tensor(
        token_selection_mask, dtype=torch.bool, device=hidden_states.device
    )
    if selection.shape != (sample_count, token_count):
        raise ValueError("token_selection_mask must match samples and tokens")
    expert_count = len(module.experts)
    if expert_count == 0:
        raise ValueError("module must contain at least one expert")
    channel_count = module.experts[0].up_proj.weight.shape[0]
    if module.gate.weight.shape != (expert_count, hidden_size):
        raise ValueError("router weight shape does not match hidden states")
    if token_metric_scale is None:
        metric_scale = torch.ones(
            (sample_count, token_count),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        metric_scale = torch.as_tensor(
            token_metric_scale,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if metric_scale.shape != selection.shape:
            raise ValueError("token_metric_scale must match samples and tokens")
        if not torch.isfinite(metric_scale).all() or (metric_scale < 0).any():
            raise ValueError("token_metric_scale must be finite and non-negative")
    selected_counts = selection.sum(dim=1)
    means = torch.zeros(
        (sample_count, expert_count, channel_count),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    routed_counts = torch.zeros(
        (sample_count, expert_count),
        dtype=torch.long,
        device=hidden_states.device,
    )
    flat_selection = selection.flatten()
    if not bool(flat_selection.any().item()):
        return means, routed_counts, selected_counts
    flat_hidden = hidden_states.reshape(-1, hidden_size)[flat_selection]
    flat_metric = metric_scale.flatten()[flat_selection]
    flat_sample_ids = (
        torch.arange(sample_count, device=hidden_states.device)
        .unsqueeze(1)
        .expand(sample_count, token_count)
        .reshape(-1)[flat_selection]
    )
    router_logits = flat_hidden @ module.gate.weight.T
    probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    top_values, top_indices = torch.topk(
        probabilities, k=int(module.top_k), dim=-1
    )
    camera_scores = top_values / top_values.sum(dim=-1, keepdim=True)
    exact_scores = camera_scores if bool(module.norm_topk_prob) else top_values
    for expert_tensor in torch.unique(top_indices):
        expert_index = int(expert_tensor.item())
        positions = torch.nonzero(top_indices == expert_index, as_tuple=False)
        token_rows = positions[:, 0]
        score_slots = positions[:, 1]
        expert_inputs = flat_hidden[token_rows]
        expert = module.experts[expert_index]
        intermediate = F.silu(expert.gate_proj(expert_inputs))
        intermediate *= expert.up_proj(expert_inputs)
        exact_routed = intermediate * exact_scores[
            token_rows, score_slots
        ].to(intermediate.dtype).unsqueeze(1)
        exact_routed = exact_routed.float()
        exact_routed *= flat_metric[token_rows].sqrt().unsqueeze(1)
        sample_ids = flat_sample_ids[token_rows]
        means[:, expert_index].index_add_(0, sample_ids, exact_routed)
        routed_counts[:, expert_index].index_add_(
            0, sample_ids, torch.ones_like(sample_ids, dtype=torch.long)
        )
    denominator = selected_counts.clamp_min(1).to(torch.float32)
    means /= denominator[:, None, None]
    return means, routed_counts, selected_counts


class Qwen3LayerStatsAccumulator:
    """Collect one MoE layer's statistics without materializing all channels.

    CAMERA L2/L-infinity energies use all routed token/expert pairs. RAMP uses
    a fixed-size random-priority reservoir per expert, which is unbiased with
    respect to stream order and bounded in memory.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        reservoir_size: int = 512,
        reservoir_seed: int = 1234,
        output_metric_scale: torch.Tensor | None = None,
        interaction_metric: str = "second_moment",
        difference_regularization: float = 0.01,
        difference_strength: float = 1.0,
        coefficient_difference_strength: float = 0.0,
    ) -> None:
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        view = resolve_moe_layer(module)
        self.module = view.mask_module
        self.router = view.router
        self.experts = view.experts
        self.expert_count = view.expert_count
        self.channel_count = view.channel_count
        self.hidden_size = view.hidden_size
        self.packed_experts = view.packed
        if self.packed_experts:
            self.device = self.experts.gate_up_proj.device
        else:
            self.device = self.experts[0].up_proj.weight.device
        self.top_k = int(
            getattr(
                self.router,
                "top_k",
                getattr(
                    self.module,
                    "top_k",
                    getattr(
                        getattr(self.router, "config", None),
                        "top_k_experts",
                        0,
                    ),
                ),
            )
        )
        if not 0 < self.top_k <= self.expert_count:
            raise ValueError("router top_k must lie inside the expert range")
        self.norm_topk_prob = bool(
            getattr(self.module, "norm_topk_prob", True)
        )
        self.activation_fn = getattr(self.experts, "act_fn", F.silu)
        if not callable(self.activation_fn):
            raise TypeError("expert activation must be callable")
        self.reservoir_capacity = reservoir_size
        if interaction_metric not in {
            "second_moment",
            "centered_difference",
            "energy_calibrated_difference",
        }:
            raise ValueError(
                "interaction_metric must be 'second_moment', "
                "'centered_difference', or 'energy_calibrated_difference'"
            )
        if not 0 <= difference_regularization <= 1:
            raise ValueError("difference_regularization must be in [0, 1]")
        self.interaction_metric = interaction_metric
        self.difference_regularization = difference_regularization
        if difference_strength < 0:
            raise ValueError("difference_strength must be non-negative")
        self.difference_strength = difference_strength
        if coefficient_difference_strength < 0:
            raise ValueError(
                "coefficient_difference_strength must be non-negative"
            )
        self.coefficient_difference_strength = coefficient_difference_strength
        if output_metric_scale is None:
            self.output_metric_scale = None
        else:
            output_metric_scale = torch.as_tensor(
                output_metric_scale, dtype=torch.float32, device=self.device
            ).flatten()
            if output_metric_scale.shape != (self.hidden_size,):
                raise ValueError(
                    "output_metric_scale must contain one value per hidden dimension"
                )
            if not torch.isfinite(output_metric_scale).all():
                raise ValueError("output_metric_scale must contain only finite values")
            self.output_metric_scale = output_metric_scale

        if self.packed_experts:
            if self.experts.gate_up_proj.device != self.experts.down_proj.device:
                raise ValueError("all packed expert weights must share a device")
        else:
            for expert in self.experts:
                if expert.up_proj.weight.device != self.device:
                    raise ValueError("all expert weights must be on the same device")

        stats_shape = (self.expert_count, self.channel_count)
        self._camera_l2_square = torch.zeros(
            stats_shape, dtype=torch.float32, device=self.device
        )
        self._camera_linf_square = torch.zeros_like(self._camera_l2_square)
        self._reservoir_values = torch.empty(
            (self.expert_count, reservoir_size, self.channel_count),
            dtype=torch.float32,
            device=self.device,
        )
        self._reservoir_router_signatures = torch.empty(
            (self.expert_count, reservoir_size, self.expert_count),
            dtype=torch.float32,
            device=self.device,
        )
        self._reservoir_keys = torch.full(
            (self.expert_count, reservoir_size),
            -torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        self._reservoir_filled = [0] * self.expert_count
        self.routed_token_counts = torch.zeros(
            self.expert_count, dtype=torch.long, device=self.device
        )
        self._routed_activation_sum = torch.zeros(
            stats_shape, dtype=torch.float32, device=self.device
        )
        self.total_input_tokens = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(reservoir_seed)

    def _update_reservoir(
        self,
        expert_index: int,
        routed_activations: torch.Tensor,
        router_signatures: torch.Tensor,
    ) -> None:
        new_count = routed_activations.shape[0]
        if new_count == 0:
            return
        if router_signatures.shape != (new_count, self.expert_count):
            raise ValueError(
                "router signatures must align with routed activations"
            )
        old_count = self._reservoir_filled[expert_index]
        new_keys = torch.rand(
            new_count,
            dtype=torch.float32,
            device=self.device,
            generator=self._generator,
        )
        candidate_keys = torch.cat(
            (self._reservoir_keys[expert_index, :old_count], new_keys)
        )
        candidate_values = torch.cat(
            (
                self._reservoir_values[expert_index, :old_count],
                routed_activations,
            )
        )
        candidate_signatures = torch.cat(
            (
                self._reservoir_router_signatures[
                    expert_index, :old_count
                ],
                router_signatures.float(),
            )
        )
        retained_count = min(self.reservoir_capacity, candidate_keys.numel())
        retained = torch.topk(
            candidate_keys, k=retained_count, largest=True, sorted=False
        ).indices
        self._reservoir_keys[expert_index, :retained_count] = candidate_keys[
            retained
        ]
        self._reservoir_values[expert_index, :retained_count] = candidate_values[
            retained
        ]
        self._reservoir_router_signatures[
            expert_index, :retained_count
        ] = candidate_signatures[retained]
        self._reservoir_filled[expert_index] = retained_count

    def _route(
        self, router_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routed = self.router(router_hidden)
        if isinstance(routed, tuple):
            if len(routed) != 3:
                raise ValueError("router tuple must contain logits, scores, and indices")
            raw_distribution, exact_scores, top_indices = routed
            if raw_distribution.shape != (
                router_hidden.shape[0],
                self.expert_count,
            ):
                raise ValueError("router distribution has an incompatible shape")
            raw_distribution = raw_distribution.float()
            row_sums = raw_distribution.sum(dim=-1)
            is_probability = bool(
                (raw_distribution >= 0).all().item()
                and torch.allclose(
                    row_sums,
                    torch.ones_like(row_sums),
                    rtol=1e-4,
                    atol=1e-5,
                )
            )
            probabilities = (
                raw_distribution
                if is_probability
                else raw_distribution.softmax(dim=-1)
            )
            exact_scores = exact_scores.float()
            top_indices = top_indices.long()
        else:
            router_logits = routed
            if router_logits.shape != (
                router_hidden.shape[0],
                self.expert_count,
            ):
                raise ValueError("router logits have an incompatible shape")
            probabilities = router_logits.float().softmax(dim=-1)
            top_values, top_indices = torch.topk(
                probabilities, k=self.top_k, dim=-1
            )
            exact_scores = (
                top_values / top_values.sum(dim=-1, keepdim=True)
                if self.norm_topk_prob
                else top_values
            )
        if top_indices.shape != (router_hidden.shape[0], self.top_k):
            raise ValueError("router indices have an incompatible shape")
        if exact_scores.shape != top_indices.shape:
            raise ValueError("router scores and indices must have the same shape")
        if (
            not torch.isfinite(exact_scores).all()
            or (exact_scores < 0).any()
        ):
            raise ValueError("router scores must be finite and non-negative")
        top_probabilities = probabilities.gather(1, top_indices)
        camera_scores = top_probabilities / top_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        return camera_scores, exact_scores, top_indices

    def _expert_intermediate(
        self, expert_index: int, expert_inputs: torch.Tensor
    ) -> torch.Tensor:
        if self.packed_experts:
            gate, up = F.linear(
                expert_inputs,
                self.experts.gate_up_proj[expert_index],
            ).chunk(2, dim=-1)
            return self.activation_fn(gate) * up
        expert = self.experts[expert_index]
        return self.activation_fn(expert.gate_proj(expert_inputs)) * expert.up_proj(
            expert_inputs
        )

    @torch.no_grad()
    def update(
        self,
        hidden_states: torch.Tensor,
        *,
        expert_hidden_states: torch.Tensor | None = None,
        token_metric_scale: torch.Tensor | None = None,
        token_selection_mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate router/expert statistics from ``[..., hidden]`` inputs."""

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden state size does not match the MoE module")
        if hidden_states.device != self.device:
            raise ValueError("hidden states and MoE module must be on the same device")
        if expert_hidden_states is None:
            expert_hidden_states = hidden_states
        if expert_hidden_states.shape != hidden_states.shape:
            raise ValueError("router and expert hidden states must have the same shape")
        if expert_hidden_states.device != self.device:
            raise ValueError("expert hidden states and weights must share a device")
        flat_router_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_expert_hidden = expert_hidden_states.reshape(-1, self.hidden_size)
        original_token_count = flat_router_hidden.shape[0]
        if original_token_count == 0:
            return
        if token_selection_mask is None:
            flat_token_selection = torch.ones(
                original_token_count, dtype=torch.bool, device=self.device
            )
        else:
            flat_token_selection = torch.as_tensor(
                token_selection_mask, dtype=torch.bool, device=self.device
            ).flatten()
            if flat_token_selection.shape != (original_token_count,):
                raise ValueError(
                    "token_selection_mask must contain one value per input token"
                )
        if token_metric_scale is None:
            flat_token_metric_scale = torch.ones(
                original_token_count, dtype=torch.float32, device=self.device
            )
        else:
            flat_token_metric_scale = torch.as_tensor(
                token_metric_scale, dtype=torch.float32, device=self.device
            ).flatten()
            if flat_token_metric_scale.shape != (original_token_count,):
                raise ValueError(
                    "token_metric_scale must contain one value per input token"
                )
            if (
                not torch.isfinite(flat_token_metric_scale).all()
                or (flat_token_metric_scale < 0).any()
            ):
                raise ValueError(
                    "token_metric_scale must contain finite non-negative values"
                )
        flat_router_hidden = flat_router_hidden[flat_token_selection]
        flat_expert_hidden = flat_expert_hidden[flat_token_selection]
        flat_token_metric_scale = flat_token_metric_scale[flat_token_selection]
        if flat_router_hidden.shape[0] == 0:
            return
        self.total_input_tokens += flat_router_hidden.shape[0]

        camera_scores, exact_scores, top_indices = self._route(
            flat_router_hidden
        )
        router_signatures = torch.zeros(
            (flat_router_hidden.shape[0], self.expert_count),
            dtype=torch.float32,
            device=self.device,
        )
        router_signatures.scatter_(1, top_indices, camera_scores)

        for expert_tensor in torch.unique(top_indices):
            expert_index = int(expert_tensor.item())
            positions = torch.nonzero(
                top_indices == expert_index, as_tuple=False
            )
            token_rows = positions[:, 0]
            score_slots = positions[:, 1]
            expert_inputs = flat_expert_hidden[token_rows]
            intermediate = self._expert_intermediate(expert_index, expert_inputs)

            camera_routed = intermediate.float() * camera_scores[
                token_rows, score_slots
            ].unsqueeze(1)
            routed_metric_scale = flat_token_metric_scale[token_rows]
            camera_squared = camera_routed.square() * routed_metric_scale.unsqueeze(1)
            self._camera_l2_square[expert_index] += camera_squared.sum(dim=0)
            self._camera_linf_square[expert_index] = torch.maximum(
                self._camera_linf_square[expert_index],
                camera_squared.amax(dim=0),
            )

            model_scores = exact_scores[token_rows, score_slots].to(
                intermediate.dtype
            )
            exact_routed = (
                intermediate * model_scores.unsqueeze(1)
            ).float()
            exact_routed *= routed_metric_scale.sqrt().unsqueeze(1)
            self._routed_activation_sum[expert_index] += exact_routed.sum(dim=0)
            self._update_reservoir(
                expert_index,
                exact_routed,
                router_signatures[token_rows],
            )
            self.routed_token_counts[expert_index] += token_rows.numel()

    def reservoir_size_for(self, expert_index: int) -> int:
        return self._reservoir_filled[expert_index]

    def router_neighborhood_samples(
        self, expert_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose activation/router samples retained by the same priorities."""

        if not 0 <= expert_index < self.expert_count:
            raise IndexError("expert_index is out of range")
        sample_count = self._reservoir_filled[expert_index]
        return (
            self._reservoir_values[expert_index, :sample_count],
            self._reservoir_router_signatures[
                expert_index, :sample_count
            ],
        )

    def local_separation_gram(
        self, expert_index: int, *, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        """Build the router-neighborhood local-separation expert Gram."""

        activations, signatures = self.router_neighborhood_samples(
            expert_index
        )
        ordinary = self.interaction_gram(expert_index)
        if activations.shape[0] == 0:
            trace: dict[str, object] = {
                "pair_count": 0,
                "neighbor_indices": [],
                "fallback": True,
                "fallback_reason": "no_routed_samples",
            }
            return (ordinary, trace) if return_trace else ordinary
        return build_router_neighborhood_local_separation_gram(
            activations,
            signatures,
            self._metric_down_projection(expert_index),
            ordinary_gram=ordinary,
            return_trace=return_trace,
        )

    def routed_activation_mean(self, expert_index: int) -> torch.Tensor:
        """Return the router-weighted phase mean with unrouted tokens as zero."""

        if not 0 <= expert_index < self.expert_count:
            raise IndexError("expert_index is out of range")
        if self.total_input_tokens == 0:
            return torch.zeros(
                self.channel_count, dtype=torch.float32, device=self.device
            )
        return self._routed_activation_sum[expert_index] / self.total_input_tokens

    def metric_down_projection(self, expert_index: int) -> torch.Tensor:
        """Expose the output-metric down basis for structured objectives."""

        if not 0 <= expert_index < self.expert_count:
            raise IndexError("expert_index is out of range")
        return self._metric_down_projection(expert_index)

    def _metric_down_projection(self, expert_index: int) -> torch.Tensor:
        if self.packed_experts:
            weight = self.experts.down_proj[expert_index].float()
        else:
            weight = self.experts[expert_index].down_proj.weight.float()
        if self.output_metric_scale is None:
            return weight
        return weight * self.output_metric_scale[:, None]

    def camera_importance(self, alpha: float = 0.9) -> torch.Tensor:
        """Return stable CAMERA importance as ``[expert, channel]``."""

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        activation_energy = (
            (1 - alpha) * self._camera_l2_square
            + alpha * self._camera_linf_square
        )
        basis_energy = torch.stack(
            [
                self._metric_down_projection(expert_index).square().sum(dim=0)
                for expert_index in range(self.expert_count)
            ]
        )
        return activation_energy * basis_energy

    def difference_importance(self) -> torch.Tensor:
        """Return routed output-difference energy for capacity allocation.

        The diagonal is the per-channel contribution to the deterministic
        all-pairs difference objective.  Expert traffic is restored after the
        routed-token mean so the score remains comparable across experts.
        """

        importance = torch.zeros(
            (self.expert_count, self.channel_count),
            dtype=torch.float32,
            device=self.device,
        )
        if self.total_input_tokens == 0:
            return importance
        for expert_index in range(self.expert_count):
            sample_count = self._reservoir_filled[expert_index]
            if sample_count == 0:
                continue
            activations = self._reservoir_values[expert_index, :sample_count]
            centered = activations - activations.mean(dim=0, keepdim=True)
            gram = build_interaction_gram(
                centered, self._metric_down_projection(expert_index)
            )
            traffic = (
                self.routed_token_counts[expert_index].float()
                / self.total_input_tokens
            )
            importance[expert_index] = gram.diagonal() * traffic
        return importance

    def interaction_gram(
        self, expert_index: int, *, normalization: str = "routed_tokens"
    ) -> torch.Tensor:
        """Build an expert Gram from its bounded routed-token reservoir."""

        if normalization not in {"routed_tokens", "all_tokens"}:
            raise ValueError(
                "normalization must be either 'routed_tokens' or 'all_tokens'"
            )
        sample_count = self._reservoir_filled[expert_index]
        if sample_count == 0:
            if normalization == "all_tokens" and self.total_input_tokens == 0:
                raise ValueError("no input tokens have been accumulated")
            return torch.zeros(
                (self.channel_count, self.channel_count),
                dtype=torch.float32,
                device=self.device,
            )
        activations = self._reservoir_values[expert_index, :sample_count]
        down_projection = self._metric_down_projection(expert_index)
        if self.interaction_metric == "centered_difference":
            gram = build_centered_difference_interaction_gram(
                activations,
                down_projection,
                regularization=self.difference_regularization,
            )
        elif self.interaction_metric == "energy_calibrated_difference":
            gram = build_energy_calibrated_difference_interaction_gram(
                activations,
                down_projection,
                strength=self.difference_strength,
            )
        else:
            gram = build_interaction_gram(activations, down_projection)
        if normalization == "routed_tokens":
            return gram
        if normalization == "all_tokens":
            if self.total_input_tokens == 0:
                raise ValueError("no input tokens have been accumulated")
            routed_count = self.routed_token_counts[expert_index].float()
            return gram * (routed_count / self.total_input_tokens)

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        """Return the configured absolute/difference amplitude-recovery Gram."""

        sample_count = self._reservoir_filled[expert_index]
        if sample_count == 0:
            return torch.zeros(
                (self.channel_count, self.channel_count),
                dtype=torch.float32,
                device=self.device,
            )
        activations = self._reservoir_values[expert_index, :sample_count]
        down_projection = self._metric_down_projection(expert_index)
        if self.coefficient_difference_strength == 0:
            return build_interaction_gram(activations, down_projection)
        return build_energy_calibrated_difference_interaction_gram(
            activations,
            down_projection,
            strength=self.coefficient_difference_strength,
        )
