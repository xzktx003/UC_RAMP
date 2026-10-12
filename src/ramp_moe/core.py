"""Numerically explicit primitives for RAMP-MoE channel selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_domain_gradients(
    domain_gradients: torch.Tensor,
) -> torch.Tensor:
    gradients = torch.as_tensor(domain_gradients, dtype=torch.float32)
    if gradients.ndim < 2 or gradients.shape[0] < 2:
        raise ValueError(
            "domain_gradients must contain at least two non-scalar domains"
        )
    if not torch.isfinite(gradients).all():
        raise ValueError("domain gradients must be finite")
    flattened = gradients.flatten(start_dim=1)
    norms = torch.linalg.vector_norm(flattened, dim=1)
    if (norms <= 0).any():
        raise ValueError("every domain gradient must have positive norm")
    shape = (gradients.shape[0],) + (1,) * (gradients.ndim - 1)
    return gradients / norms.reshape(shape)


def minimum_norm_simplex_gradient(
    domain_gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the scale-invariant minimum-norm convex domain gradient.

    The simplex quadratic program is solved without an optimizer or learned
    weights.  Because IDEA-066 has five domains, enumerating every non-empty
    active set is both exact and tiny.  Singular faces use the minimum-norm
    KKT solution; equal-objective ties prefer the larger, symmetric support.
    """

    normalized = _normalize_domain_gradients(domain_gradients)
    flattened = normalized.flatten(start_dim=1).to(dtype=torch.float64)
    gram = flattened @ flattened.T
    domain_count = gram.shape[0]
    machine = torch.finfo(torch.float64).eps
    best_objective: float | None = None
    best_weights: torch.Tensor | None = None
    best_support = -1
    for active_bits in range(1, 1 << domain_count):
        active = [
            index
            for index in range(domain_count)
            if active_bits & (1 << index)
        ]
        active_gram = gram[active][:, active]
        size = len(active)
        kkt = torch.zeros(
            (size + 1, size + 1),
            dtype=torch.float64,
            device=gram.device,
        )
        kkt[:size, :size] = 2 * active_gram
        kkt[:size, size] = 1
        kkt[size, :size] = 1
        rhs = torch.zeros(size + 1, dtype=torch.float64, device=gram.device)
        rhs[size] = 1
        solution = torch.linalg.lstsq(kkt, rhs).solution[:size]
        feasibility_tolerance = machine * max(size, 1) * 64
        if (
            not torch.isfinite(solution).all()
            or float(solution.min().item()) < -feasibility_tolerance
        ):
            continue
        solution = solution.clamp_min(0)
        total = solution.sum()
        if total <= 0:
            continue
        solution /= total
        objective = float((solution @ active_gram @ solution).item())
        objective_tolerance = machine * max(abs(objective), 1.0) * 128
        better = best_objective is None or objective < (
            best_objective - objective_tolerance
        )
        tied_with_larger_support = (
            best_objective is not None
            and abs(objective - best_objective) <= objective_tolerance
            and size > best_support
        )
        if better or tied_with_larger_support:
            weights = torch.zeros(
                domain_count, dtype=torch.float64, device=gram.device
            )
            weights[active] = solution
            best_objective = objective
            best_weights = weights
            best_support = size
    if best_weights is None:
        raise RuntimeError("failed to solve the domain-gradient simplex")
    weights = best_weights.to(dtype=torch.float32)
    combined = torch.einsum("d,d...->...", weights, normalized)
    return weights, combined


def project_pareto_taylor_swap_prefix(
    domain_gradients: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    combination_weights: torch.Tensor | None = None,
    loop_domain: int = -1,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Project end-to-end Taylor gradients onto one coordinated swap field.

    Each layer/expert proposes its best same-width 1-swap under the MGDA
    gradient.  Proposals are ordered once, then a *global* prefix is selected
    only when every normalized domain has non-positive cumulative first-order
    change and the designated loop domain strictly improves.  This permits
    cross-expert compensation without a tunable domain weight or swap count.
    """

    mask = torch.as_tensor(base_mask)
    if mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError("base_mask must be boolean [layers, experts, channels]")
    gradients = torch.as_tensor(
        domain_gradients, dtype=torch.float32, device=mask.device
    )
    if gradients.ndim != 4 or gradients.shape[1:] != mask.shape:
        raise ValueError(
            "domain_gradients must have shape [domains, layers, experts, channels]"
        )
    normalized = _normalize_domain_gradients(gradients)
    domain_count = normalized.shape[0]
    resolved_loop_domain = loop_domain % domain_count
    if not 0 <= resolved_loop_domain < domain_count:
        raise ValueError("loop_domain is out of range")
    if combination_weights is None:
        weights, combined = minimum_norm_simplex_gradient(normalized)
    else:
        weights = torch.as_tensor(
            combination_weights, dtype=torch.float32, device=mask.device
        ).flatten()
        if (
            weights.numel() != domain_count
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.isclose(weights.sum(), torch.tensor(1.0, device=mask.device))
        ):
            raise ValueError(
                "combination_weights must be a finite simplex vector"
            )
        combined = torch.einsum("d,d...->...", weights, normalized)

    proposals: list[dict[str, object]] = []
    epsilon = torch.finfo(torch.float32).eps
    for layer_index in range(mask.shape[0]):
        for expert_index in range(mask.shape[1]):
            expert_mask = mask[layer_index, expert_index]
            retained = torch.nonzero(expert_mask, as_tuple=False).flatten()
            pruned = torch.nonzero(~expert_mask, as_tuple=False).flatten()
            if retained.numel() == 0 or pruned.numel() == 0:
                continue
            expert_combined = combined[layer_index, expert_index]
            outgoing = retained[
                torch.argmax(expert_combined[retained])
            ]
            incoming = pruned[
                torch.argmin(expert_combined[pruned])
            ]
            combined_delta = expert_combined[incoming] - expert_combined[outgoing]
            proposal_tolerance = epsilon * max(
                abs(float(expert_combined[incoming].item())),
                abs(float(expert_combined[outgoing].item())),
                1.0,
            )
            if float(combined_delta.item()) >= -proposal_tolerance:
                continue
            deltas = (
                normalized[:, layer_index, expert_index, incoming]
                - normalized[:, layer_index, expert_index, outgoing]
            )
            proposals.append(
                {
                    "layer": layer_index,
                    "expert": expert_index,
                    "outgoing": int(outgoing.item()),
                    "incoming": int(incoming.item()),
                    "combined_delta": float(combined_delta.item()),
                    "domain_deltas_tensor": deltas,
                }
            )
    proposals.sort(
        key=lambda item: (
            item["combined_delta"],
            item["layer"],
            item["expert"],
            item["outgoing"],
            item["incoming"],
        )
    )

    cumulative = torch.zeros(
        domain_count, dtype=torch.float32, device=mask.device
    )
    cumulative_combined = 0.0
    best_prefix = 0
    best_key: tuple[float, float, int] | None = None
    best_domain_deltas = cumulative.clone()
    for prefix_index, proposal in enumerate(proposals, start=1):
        cumulative += proposal["domain_deltas_tensor"]
        cumulative_combined += float(proposal["combined_delta"])
        tolerance = epsilon * cumulative.abs().sum().clamp_min(1)
        feasible = bool((cumulative <= tolerance).all().item()) and bool(
            (cumulative[resolved_loop_domain] < -tolerance).item()
        )
        if not feasible:
            continue
        key = (
            float(cumulative[resolved_loop_domain].item()),
            cumulative_combined,
            prefix_index,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_prefix = prefix_index
            best_domain_deltas = cumulative.clone()

    refined = mask.clone()
    selected_records: list[dict[str, object]] = []
    for proposal in proposals[:best_prefix]:
        layer_index = int(proposal["layer"])
        expert_index = int(proposal["expert"])
        outgoing = int(proposal["outgoing"])
        incoming = int(proposal["incoming"])
        refined[layer_index, expert_index, outgoing] = False
        refined[layer_index, expert_index, incoming] = True
        selected_records.append(
            {
                "layer": layer_index,
                "expert": expert_index,
                "outgoing": outgoing,
                "incoming": incoming,
                "combined_delta": float(proposal["combined_delta"]),
                "domain_deltas": [
                    float(value)
                    for value in proposal["domain_deltas_tensor"].tolist()
                ],
            }
        )
    if not torch.equal(refined.sum(dim=-1), mask.sum(dim=-1)):
        raise RuntimeError("Pareto Taylor projection changed expert widths")
    trace: dict[str, object] = {
        "status": "selected" if best_prefix > 0 else "no_pareto_prefix",
        "proposal_count": len(proposals),
        "selected_swaps": best_prefix,
        "combination_weights": [float(value) for value in weights.tolist()],
        "domain_deltas": [
            float(value) for value in best_domain_deltas.tolist()
        ],
        "combined_delta": (
            float(best_key[1]) if best_key is not None else 0.0
        ),
        "selected": selected_records,
    }
    return (refined, trace) if return_trace else refined


def find_minimal_vector_feasible_pair(
    proposal_domain_deltas: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    clean_domains: tuple[int, ...],
    strict_domains: tuple[int, ...],
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Find an exact cardinality-one or cardinality-two Pareto witness.

    Unlike a frozen proposal prefix, this search may combine any two atomic
    exchanges from different groups.  It is intentionally a bounded,
    model-free witness search rather than a general subset optimizer: finding
    a pair proves that vector compensation expands the prefix feasible set,
    while a null result says nothing about larger subsets.
    """

    deltas = torch.as_tensor(proposal_domain_deltas, dtype=torch.float32)
    if deltas.ndim != 2 or deltas.shape[1] < 2:
        raise ValueError("proposal_domain_deltas must be [proposals, domains]")
    if not torch.isfinite(deltas).all():
        raise ValueError("proposal domain deltas must be finite")
    groups = torch.as_tensor(group_ids, dtype=torch.long, device=deltas.device)
    if groups.ndim != 1 or groups.numel() != deltas.shape[0]:
        raise ValueError("group_ids must contain one value per proposal")
    if not clean_domains or not strict_domains:
        raise ValueError("clean_domains and strict_domains must be non-empty")
    all_domains = clean_domains + strict_domains
    if (
        len(set(all_domains)) != len(all_domains)
        or min(all_domains) < 0
        or max(all_domains) >= deltas.shape[1]
    ):
        raise ValueError("domain constraints must be unique and in range")

    clean = torch.tensor(clean_domains, dtype=torch.long, device=deltas.device)
    strict = torch.tensor(strict_domains, dtype=torch.long, device=deltas.device)
    epsilon = torch.finfo(torch.float32).eps

    def feasible(values: torch.Tensor) -> torch.Tensor:
        tolerance = epsilon * values.abs().sum(dim=-1).clamp_min(1)
        clean_safe = (
            values.index_select(-1, clean) <= tolerance.unsqueeze(-1)
        ).all(dim=-1)
        strict_gain = (
            values.index_select(-1, strict) < -tolerance.unsqueeze(-1)
        ).all(dim=-1)
        return clean_safe & strict_gain

    def objective(values: torch.Tensor, indices: tuple[int, ...]) -> tuple[object, ...]:
        return (
            float(values.index_select(0, strict).max().item()),
            float(values.index_select(0, clean).max().item()),
            float(values.sum().item()),
            *indices,
        )

    singleton_mask = feasible(deltas)
    singleton_indices = torch.nonzero(singleton_mask, as_tuple=False).flatten()
    if singleton_indices.numel() > 0:
        selected_index = min(
            (int(index.item()) for index in singleton_indices),
            key=lambda index: objective(deltas[index], (index,)),
        )
        selected = torch.tensor(
            [selected_index], dtype=torch.long, device=deltas.device
        )
        selected_delta = deltas[selected_index]
        trace = {
            "status": "selected_singleton",
            "cardinality": 1,
            "selected_indices": [selected_index],
            "domain_deltas": [float(value) for value in selected_delta.tolist()],
        }
        return (selected, trace) if return_trace else selected

    proposal_count = deltas.shape[0]
    best_key: tuple[object, ...] | None = None
    best_pair: tuple[int, int] | None = None
    best_delta: torch.Tensor | None = None
    right_indices = torch.arange(proposal_count, device=deltas.device)
    block_size = 128
    for start in range(0, proposal_count, block_size):
        stop = min(start + block_size, proposal_count)
        left_indices = torch.arange(start, stop, device=deltas.device)
        pair_deltas = deltas[left_indices, None, :] + deltas[None, :, :]
        eligible = right_indices.unsqueeze(0) > left_indices.unsqueeze(1)
        eligible &= groups[left_indices].unsqueeze(1) != groups.unsqueeze(0)
        eligible &= feasible(pair_deltas)
        locations = torch.nonzero(eligible, as_tuple=False)
        if locations.numel() == 0:
            continue
        for local_left, right in locations.tolist():
            left = start + int(local_left)
            right = int(right)
            delta = pair_deltas[local_left, right]
            key = objective(delta, (left, right))
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (left, right)
                best_delta = delta.clone()

    if best_pair is None or best_delta is None:
        selected = torch.empty(0, dtype=torch.long, device=deltas.device)
        trace = {
            "status": "no_feasible_pair",
            "cardinality": 0,
            "selected_indices": [],
            "domain_deltas": [0.0] * deltas.shape[1],
        }
        return (selected, trace) if return_trace else selected

    selected = torch.tensor(best_pair, dtype=torch.long, device=deltas.device)
    trace = {
        "status": "selected_pair",
        "cardinality": 2,
        "selected_indices": list(best_pair),
        "domain_deltas": [float(value) for value in best_delta.tolist()],
    }
    return (selected, trace) if return_trace else selected


def apply_taylor_swap_prefix(
    base_mask: torch.Tensor,
    selected_records: list[dict[str, object]],
    prefix_length: int,
) -> torch.Tensor:
    """Replay an exact prefix of audited same-width Taylor swaps."""

    mask = torch.as_tensor(base_mask)
    if mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError("base_mask must be boolean [layers, experts, channels]")
    if not 0 <= prefix_length <= len(selected_records):
        raise ValueError("prefix_length is outside the selected swap records")
    refined = mask.clone()
    seen_experts: set[tuple[int, int]] = set()
    for record in selected_records[:prefix_length]:
        try:
            layer = int(record["layer"])
            expert = int(record["expert"])
            outgoing = int(record["outgoing"])
            incoming = int(record["incoming"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Taylor swap record is incomplete") from error
        if (
            not 0 <= layer < mask.shape[0]
            or not 0 <= expert < mask.shape[1]
            or not 0 <= outgoing < mask.shape[2]
            or not 0 <= incoming < mask.shape[2]
            or outgoing == incoming
        ):
            raise ValueError("Taylor swap record contains an invalid index")
        identity = (layer, expert)
        if identity in seen_experts:
            raise ValueError("Taylor prefix may swap each expert at most once")
        seen_experts.add(identity)
        if not bool(refined[layer, expert, outgoing]) or bool(
            refined[layer, expert, incoming]
        ):
            raise ValueError("Taylor swap record does not match the base mask")
        refined[layer, expert, outgoing] = False
        refined[layer, expert, incoming] = True
    if not torch.equal(refined.sum(dim=-1), mask.sum(dim=-1)):
        raise RuntimeError("Taylor prefix changed expert widths")
    return refined


def halving_prefix_lengths(selected_swaps: int) -> list[int]:
    """Return the fixed discrete backtracking schedule down to one swap."""

    if selected_swaps < 0:
        raise ValueError("selected_swaps must be non-negative")
    schedule: list[int] = []
    current = selected_swaps
    while current > 0:
        schedule.append(current)
        current //= 2
    return schedule


def _validate_contribution_inputs(
    routed_activations: torch.Tensor, down_weight: torch.Tensor
) -> None:
    if routed_activations.ndim != 2:
        raise ValueError("routed_activations must have shape [tokens, channels]")
    if down_weight.ndim != 2:
        raise ValueError("down_weight must have shape [output_size, channels]")
    if routed_activations.shape[0] == 0:
        raise ValueError("routed_activations must contain at least one token")
    if routed_activations.shape[1] != down_weight.shape[1]:
        raise ValueError("activation and down-projection channel counts must match")


def _validate_gram(gram: torch.Tensor) -> None:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    if gram.shape[0] == 0:
        raise ValueError("gram must contain at least one channel")
    if not torch.allclose(gram, gram.T, rtol=1e-5, atol=1e-6):
        raise ValueError("gram must be symmetric")


def _validate_keep_count(gram: torch.Tensor, keep_count: int) -> None:
    _validate_gram(gram)
    if not 1 <= keep_count <= gram.shape[0]:
        raise ValueError(
            f"keep_count must be in [1, {gram.shape[0]}], got {keep_count}"
        )


def _normalize_keep_indices(
    keep_indices: torch.Tensor, channel_count: int, device: torch.device
) -> torch.Tensor:
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=device).flatten()
    if keep.numel() == 0:
        raise ValueError("keep_indices must contain at least one channel")
    if keep.unique().numel() != keep.numel():
        raise ValueError("keep_indices must not contain duplicates")
    if keep.min() < 0 or keep.max() >= channel_count:
        raise ValueError("keep_indices contains an out-of-range channel")
    return keep


def build_interaction_gram(
    routed_activations: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    """Build ``H=(Z.T@Z/N) * (W_down.T@W_down)`` in FP32."""

    _validate_contribution_inputs(routed_activations, down_weight)
    activations_fp32 = routed_activations.float()
    down_fp32 = down_weight.float()
    activation_gram = activations_fp32.T @ activations_fp32
    activation_gram /= activations_fp32.shape[0]
    weight_gram = down_fp32.T @ down_fp32
    return activation_gram * weight_gram


def build_router_neighborhood_local_separation_gram(
    routed_activations: torch.Tensor,
    router_signatures: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    ordinary_gram: torch.Tensor | None = None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Build a trace-matched Gram from router-nearest activation differences.

    Every routed activation is paired with the other sample having maximum
    cosine similarity in sparse TopK-router space. ``torch.argmax`` provides
    the fixed lowest-index tie break. Fewer than two samples or a degenerate
    local objective falls back exactly to the ordinary interaction Gram.
    """

    _validate_contribution_inputs(routed_activations, down_weight)
    signatures = torch.as_tensor(
        router_signatures,
        dtype=torch.float32,
        device=routed_activations.device,
    )
    if signatures.ndim != 2:
        raise ValueError("router_signatures must have shape [tokens, experts]")
    if signatures.shape[0] != routed_activations.shape[0]:
        raise ValueError("router signatures and activations must share tokens")
    if signatures.shape[1] == 0:
        raise ValueError("router_signatures must contain at least one expert")
    if not torch.isfinite(signatures).all() or (signatures < 0).any():
        raise ValueError("router signatures must be finite and non-negative")

    ordinary = (
        build_interaction_gram(routed_activations, down_weight)
        if ordinary_gram is None
        else torch.as_tensor(
            ordinary_gram,
            dtype=torch.float32,
            device=routed_activations.device,
        )
    )
    _validate_gram(ordinary)
    if ordinary.shape[0] != routed_activations.shape[1]:
        raise ValueError("ordinary_gram channel count does not match activations")

    sample_count = routed_activations.shape[0]
    trace: dict[str, object] = {
        "pair_count": 0,
        "neighbor_indices": [],
        "fallback": True,
        "fallback_reason": "insufficient_samples",
    }
    if sample_count < 2:
        return (ordinary, trace) if return_trace else ordinary

    norms = torch.linalg.vector_norm(signatures, dim=1, keepdim=True)
    if bool((norms <= 0).any().item()):
        trace["fallback_reason"] = "zero_router_signature"
        return (ordinary, trace) if return_trace else ordinary
    normalized = signatures / norms
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    neighbor_indices = similarity.argmax(dim=1)
    local_differences = routed_activations.float() - routed_activations.float()[
        neighbor_indices
    ]
    local = build_interaction_gram(local_differences, down_weight)
    local_trace = local.trace()
    ordinary_trace = ordinary.trace()
    tolerance = torch.finfo(torch.float32).eps * ordinary_trace.abs().clamp_min(1)
    trace.update(
        {
            "pair_count": sample_count,
            "neighbor_indices": neighbor_indices.detach().cpu().tolist(),
            "local_trace_before_match": float(local_trace.item()),
            "ordinary_trace": float(ordinary_trace.item()),
        }
    )
    if not torch.isfinite(local).all() or local_trace <= tolerance:
        trace["fallback_reason"] = "degenerate_local_trace"
        return (ordinary, trace) if return_trace else ordinary

    matched = local * (ordinary_trace / local_trace)
    trace["fallback"] = False
    trace["fallback_reason"] = None
    return (matched, trace) if return_trace else matched


def universal_router_neighborhood_objective(
    ordinary_grams: tuple[torch.Tensor, ...],
    local_grams: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Blend the fixed universal anchor with its local-separation view."""

    if len(ordinary_grams) != 4 or len(local_grams) != 4:
        raise ValueError("universal router objective requires exactly four domains")
    reference = ordinary_grams[0]
    _validate_gram(reference)
    for gram in (*ordinary_grams[1:], *local_grams):
        _validate_gram(gram)
        if gram.shape != reference.shape or gram.device != reference.device:
            raise ValueError("universal router grams must share shape and device")
    specialist_weight = 0.1 / 3
    anchor = 0.9 * ordinary_grams[0] + specialist_weight * sum(
        ordinary_grams[1:]
    )
    local_anchor = 0.9 * local_grams[0] + specialist_weight * sum(
        local_grams[1:]
    )
    anchor_trace = anchor.trace()
    local_trace = local_anchor.trace()
    tolerance = torch.finfo(anchor.dtype).eps * anchor_trace.abs().clamp_min(1)
    if not torch.isfinite(local_anchor).all() or local_trace <= tolerance:
        return anchor
    matched_local = local_anchor * (anchor_trace / local_trace)
    return 0.5 * (anchor + matched_local)


def orthogonal_recovery_residuals(
    loop_means: torch.Tensor,
    gold_means: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Remove each loop-parallel component from its paired gold phase."""

    loop = torch.as_tensor(loop_means, dtype=torch.float32)
    gold = torch.as_tensor(
        gold_means, dtype=torch.float32, device=loop.device
    )
    down = torch.as_tensor(
        down_weight, dtype=torch.float32, device=loop.device
    )
    if loop.ndim != 2 or gold.shape != loop.shape or loop.shape[0] == 0:
        raise ValueError(
            "loop and gold means must share non-empty [pairs, channels] shape"
        )
    if down.ndim != 2 or down.shape[1] != loop.shape[1]:
        raise ValueError(
            "down_weight must have shape [output_size, channels]"
        )
    if not torch.isfinite(loop).all() or not torch.isfinite(gold).all():
        raise ValueError("loop and gold means must be finite")
    metric = down.T @ down
    numerator = torch.einsum("pc,cd,pd->p", loop, metric, gold)
    denominator = torch.einsum("pc,cd,pd->p", loop, metric, loop)
    epsilon = torch.finfo(torch.float32).eps
    coefficients = torch.where(
        denominator > epsilon,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return gold - coefficients.unsqueeze(1) * loop


def build_loop_orthogonal_importance(
    loop_means: torch.Tensor,
    gold_means: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    anchor_importance: torch.Tensor,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Geometrically reveal recovery-only channels against a fixed anchor."""

    anchor = torch.as_tensor(
        anchor_importance,
        dtype=torch.float32,
        device=torch.as_tensor(loop_means).device,
    ).flatten()
    if anchor.shape != (torch.as_tensor(loop_means).shape[-1],):
        raise ValueError("anchor_importance must contain one value per channel")
    if not torch.isfinite(anchor).all() or (anchor < 0).any():
        raise ValueError("anchor_importance must be finite and non-negative")
    residuals = orthogonal_recovery_residuals(
        loop_means, gold_means, down_weight
    )
    recovery_gram = build_interaction_gram(residuals, down_weight)
    recovery = recovery_gram.diagonal().clamp_min(0)
    recovery_mass = recovery.sum()
    anchor_mass = anchor.sum()
    trace: dict[str, object] = {
        "pair_count": residuals.shape[0],
        "anchor_mass": float(anchor_mass.item()),
        "recovery_mass_before_match": float(recovery_mass.item()),
        "fallback": True,
        "fallback_reason": "zero_orthogonal_recovery",
    }
    if (
        not torch.isfinite(recovery).all()
        or recovery_mass <= 0
        or anchor_mass <= 0
    ):
        return (anchor, trace) if return_trace else anchor
    matched = recovery * (anchor_mass / recovery_mass)
    importance = torch.sqrt(anchor * matched)
    trace["fallback"] = False
    trace["fallback_reason"] = None
    trace["matched_recovery_mass"] = float(matched.sum().item())
    return (importance, trace) if return_trace else importance


def build_trajectory_transition_components(
    anchor_gram: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    problem_mean: torch.Tensor | None,
    solution_mean: torch.Tensor | None,
    termination_mean: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return the anchor and separately trace-matched ordered transitions.

    Missing or degenerate phase observations remain ``None`` so callers can
    constrain only transitions that were actually observed.
    """

    _validate_gram(anchor_gram)
    if down_weight.ndim != 2 or down_weight.shape[1] != anchor_gram.shape[0]:
        raise ValueError("down_weight must match the anchor channel count")
    channel_count = anchor_gram.shape[0]
    phases = (problem_mean, solution_mean, termination_mean)
    for phase in phases:
        if phase is None:
            continue
        phase_tensor = torch.as_tensor(
            phase, dtype=torch.float32, device=anchor_gram.device
        ).flatten()
        if phase_tensor.shape != (channel_count,) or not torch.isfinite(
            phase_tensor
        ).all():
            raise ValueError("phase means must be finite channel vectors")

    anchor = anchor_gram.float()
    anchor_trace = anchor.trace()
    tolerance = torch.finfo(anchor.dtype).eps * anchor_trace.abs().clamp_min(1)
    if anchor_trace <= tolerance:
        return anchor, None, None
    down = down_weight.to(device=anchor.device, dtype=torch.float32)
    weight_gram = down.T @ down
    transitions: list[torch.Tensor | None] = []
    for left, right in zip(phases, phases[1:]):
        if left is None or right is None:
            transitions.append(None)
            continue
        delta = right.to(anchor.device, torch.float32) - left.to(
            anchor.device, torch.float32
        )
        transition = torch.outer(delta, delta) * weight_gram
        transition_trace = transition.trace()
        if transition_trace <= tolerance:
            transitions.append(None)
            continue
        transitions.append(transition * (anchor_trace / transition_trace))
    return anchor, transitions[0], transitions[1]


def build_trajectory_transition_gram(
    anchor_gram: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    problem_mean: torch.Tensor | None,
    solution_mean: torch.Tensor | None,
    termination_mean: torch.Tensor | None,
) -> torch.Tensor:
    """Blend observed ordered phase transitions with an absolute Gram anchor."""

    components = build_trajectory_transition_components(
        anchor_gram,
        down_weight,
        problem_mean=problem_mean,
        solution_mean=solution_mean,
        termination_mean=termination_mean,
    )
    observed = [component for component in components if component is not None]
    return torch.stack(observed).mean(dim=0)


def build_centered_difference_interaction_gram(
    routed_activations: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    regularization: float = 0.01,
) -> torch.Tensor:
    """Blend ordinary and centered routed-channel contribution Grams.

    Centering is the deterministic expectation-equivalent form of random
    activation differences, up to a common factor.  The difference Gram is
    trace-matched *after* applying the down-projection output metric so the
    blend preserves ordinary output-contribution energy.  A degenerate
    centered Gram falls back to the ordinary objective.
    """

    if not 0 <= regularization <= 1:
        raise ValueError("regularization must be in [0, 1]")
    ordinary = build_interaction_gram(routed_activations, down_weight)
    activations_fp32 = routed_activations.float()
    centered = activations_fp32 - activations_fp32.mean(dim=0, keepdim=True)
    difference = build_interaction_gram(centered, down_weight)
    difference_trace = difference.trace()
    ordinary_trace = ordinary.trace()
    tolerance = torch.finfo(difference.dtype).eps * ordinary_trace.abs().clamp_min(1)
    if difference_trace <= tolerance:
        return ordinary
    trace_ratio = ordinary_trace / difference_trace
    return regularization * ordinary + (1 - regularization) * trace_ratio * difference


def build_energy_calibrated_difference_interaction_gram(
    routed_activations: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Add centered contribution energy without trace amplification.

    Unlike the trace-matched objective, this shrinkage form preserves the
    empirical routed-variance scale.  ``strength=0`` is exactly the ordinary
    second-moment interaction Gram.
    """

    if strength < 0:
        raise ValueError("strength must be non-negative")
    ordinary = build_interaction_gram(routed_activations, down_weight)
    if strength == 0:
        return ordinary
    activations_fp32 = routed_activations.float()
    centered = activations_fp32 - activations_fp32.mean(dim=0, keepdim=True)
    difference = build_interaction_gram(centered, down_weight)
    return ordinary + strength * difference


def build_explicit_gram(
    routed_activations: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    """Build the reference ``Q.T@Q/N`` by materializing channel outputs."""

    _validate_contribution_inputs(routed_activations, down_weight)
    contributions = torch.einsum(
        "nc,oc->noc", routed_activations.float(), down_weight.float()
    )
    explicit_q = contributions.reshape(-1, contributions.shape[-1])
    return explicit_q.T @ explicit_q / routed_activations.shape[0]


def gram_reconstruction_error(
    gram: torch.Tensor, keep_indices: torch.Tensor
) -> torch.Tensor:
    """Return the exact per-token squared error ``p.T@H@p``."""

    _validate_gram(gram)
    raw_keep = torch.as_tensor(
        keep_indices, dtype=torch.long, device=gram.device
    ).flatten()
    if raw_keep.numel() == 0:
        pruned = torch.ones(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        return pruned @ gram @ pruned
    keep = _normalize_keep_indices(raw_keep, gram.shape[0], gram.device)
    pruned = torch.ones(gram.shape[0], dtype=gram.dtype, device=gram.device)
    pruned[keep] = 0
    return pruned @ gram @ pruned


def recover_channel_coefficients(
    gram: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    ridge_ratio: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Solve regularized retained-channel coefficients from an exact Gram."""

    _validate_gram(gram)
    keep = _normalize_keep_indices(keep_indices, gram.shape[0], gram.device)
    if ridge_ratio <= 0:
        raise ValueError("ridge_ratio must be positive")
    if not 0 <= clip_min <= clip_max:
        raise ValueError("coefficient clip bounds must satisfy 0 <= min <= max")
    if keep.numel() == gram.shape[0]:
        return torch.ones(keep.numel(), dtype=gram.dtype, device=gram.device)

    retained_gram = gram[keep][:, keep]
    scale = retained_gram.diag().mean().clamp_min(0)
    if scale <= torch.finfo(gram.dtype).eps:
        return torch.ones(keep.numel(), dtype=gram.dtype, device=gram.device)
    ridge = ridge_ratio * scale
    identity = torch.eye(
        keep.numel(), dtype=gram.dtype, device=gram.device
    )
    target = gram[keep].sum(dim=1) + ridge
    coefficients = torch.linalg.solve(
        retained_gram + ridge * identity, target
    )
    return coefficients.clamp(min=clip_min, max=clip_max)


def direct_reconstruction_error(
    routed_activations: torch.Tensor,
    down_weight: torch.Tensor,
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute reconstruction error from full and retained expert outputs."""

    _validate_contribution_inputs(routed_activations, down_weight)
    keep = _normalize_keep_indices(
        keep_indices, routed_activations.shape[1], routed_activations.device
    )
    activations_fp32 = routed_activations.float()
    down_fp32 = down_weight.float()
    full_output = activations_fp32 @ down_fp32.T
    retained_output = activations_fp32[:, keep] @ down_fp32[:, keep].T
    return (full_output - retained_output).square().sum() / activations_fp32.shape[0]


def diagonal_keep_indices(gram: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Retain the channels with the largest independent output energies."""

    _validate_keep_count(gram, keep_count)
    return torch.argsort(gram.diag(), descending=True, stable=True)[:keep_count]


def forward_residual_keep_indices(
    gram: torch.Tensor, keep_count: int
) -> torch.Tensor:
    """Greedily maximize the exact fixed-coefficient marginal residual gain."""

    _validate_keep_count(gram, keep_count)
    channel_count = gram.shape[0]
    selected = torch.zeros(channel_count, dtype=torch.bool, device=gram.device)
    residual_dot = gram.sum(dim=1).clone()
    choices: list[torch.Tensor] = []

    for _ in range(keep_count):
        gains = 2 * residual_dot - gram.diag()
        gains[selected] = -torch.inf
        choice = torch.argmax(gains)
        choices.append(choice)
        selected[choice] = True
        residual_dot -= gram[:, choice]

    return torch.stack(choices)


def reweight_gram_with_importance(
    gram: torch.Tensor,
    importance: torch.Tensor,
    *,
    interaction_strength: float,
) -> torch.Tensor:
    """Combine a robust diagonal importance with Gram correlations.

    ``interaction_strength=0`` recovers independent importance ranking. At a
    positive strength, off-diagonal entries preserve the correlation structure
    of ``gram`` while using ``importance`` as the exact diagonal scale.
    """

    _validate_gram(gram)
    importance = torch.as_tensor(
        importance, dtype=gram.dtype, device=gram.device
    ).flatten()
    if importance.shape != (gram.shape[0],):
        raise ValueError("importance must contain one value per channel")
    if not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("importance must contain finite non-negative values")
    if not 0 <= interaction_strength <= 1:
        raise ValueError("interaction_strength must be in [0, 1]")

    gram_diagonal = gram.diag().clamp_min(0)
    denominator = torch.sqrt(
        gram_diagonal[:, None] * gram_diagonal[None, :]
    )
    correlation = torch.zeros_like(gram)
    valid = denominator > 0
    correlation[valid] = gram[valid] / denominator[valid]
    positive_importance = importance > 0
    correlation.diagonal().copy_(positive_importance.to(gram.dtype))

    robust_scale = torch.sqrt(importance)
    correlated = correlation * (
        robust_scale[:, None] * robust_scale[None, :]
    )
    result = correlated * interaction_strength
    result.diagonal().copy_(importance)
    return result


def swap_refine_keep_indices(
    gram: torch.Tensor,
    initial_keep_indices: torch.Tensor,
    *,
    max_swaps: int,
) -> torch.Tensor:
    """Apply deterministic improving 1-swaps to an existing retained set."""

    _validate_gram(gram)
    if max_swaps < 0:
        raise ValueError("max_swaps must be non-negative")
    keep = _normalize_keep_indices(
        initial_keep_indices, gram.shape[0], gram.device
    )
    selected = torch.zeros(
        gram.shape[0], dtype=torch.bool, device=gram.device
    )
    selected[keep] = True
    if keep.numel() == gram.shape[0] or max_swaps == 0:
        return torch.nonzero(selected, as_tuple=False).flatten()

    diagonal = gram.diag()
    for _ in range(max_swaps):
        retained = torch.nonzero(selected, as_tuple=False).flatten()
        pruned = torch.nonzero(~selected, as_tuple=False).flatten()
        pruned_indicator = (~selected).to(gram.dtype)
        residual_dot = gram @ pruned_indicator
        delta = (
            2 * (residual_dot[retained, None] - residual_dot[pruned][None, :])
            + diagonal[retained, None]
            + diagonal[pruned][None, :]
            - 2 * gram[retained[:, None], pruned[None, :]]
        )
        flat_choice = torch.argmin(delta)
        best_delta = delta.flatten()[flat_choice]
        current_error = pruned_indicator @ gram @ pruned_indicator
        tolerance = torch.finfo(gram.dtype).eps * current_error.abs().clamp_min(1)
        if best_delta >= -tolerance:
            break
        row = torch.div(flat_choice, pruned.numel(), rounding_mode="floor")
        column = flat_choice % pruned.numel()
        selected[retained[row]] = False
        selected[pruned[column]] = True

    return torch.nonzero(selected, as_tuple=False).flatten()


def guarded_swap_refine_keep_indices(
    proposal_gram: torch.Tensor,
    constraint_grams: tuple[torch.Tensor, ...],
    initial_keep_indices: torch.Tensor,
    *,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Accept the proposal Gram's best 1-swap only when every view is safe.

    The guard deliberately does not search for a different admissible swap.
    Consequently its accepted exchanges are a strict subset of the unguarded
    single-swap proposal, which isolates the effect of the safety constraint.
    """

    _validate_gram(proposal_gram)
    if not constraint_grams:
        raise ValueError("guarded refinement requires at least one constraint gram")
    for gram in constraint_grams:
        _validate_gram(gram)
        if gram.shape != proposal_gram.shape:
            raise ValueError("constraint grams must match the proposal shape")
        if gram.device != proposal_gram.device:
            raise ValueError("constraint grams must share the proposal device")

    raw_keep = torch.as_tensor(
        initial_keep_indices, dtype=torch.long, device=proposal_gram.device
    ).flatten()
    empty_trace: dict[str, object] = {
        "proposed": False,
        "accepted": False,
        "outgoing_channel": None,
        "incoming_channel": None,
        "constraint_deltas": [],
        "constraint_tolerances": [],
    }
    if raw_keep.numel() == 0 or raw_keep.numel() == proposal_gram.shape[0]:
        return (raw_keep, empty_trace) if return_trace else raw_keep
    keep = _normalize_keep_indices(
        raw_keep, proposal_gram.shape[0], proposal_gram.device
    )
    proposed = swap_refine_keep_indices(
        proposal_gram, keep, max_swaps=1
    )
    initial_set = set(keep.cpu().tolist())
    proposed_set = set(proposed.cpu().tolist())
    if proposed_set == initial_set:
        return (keep, empty_trace) if return_trace else keep

    outgoing = next(iter(initial_set - proposed_set))
    incoming = next(iter(proposed_set - initial_set))
    deltas: list[float] = []
    tolerances: list[float] = []
    safe = True
    strictly_improving = False
    for gram in constraint_grams:
        current_error = gram_reconstruction_error(gram.float(), keep)
        proposed_error = gram_reconstruction_error(gram.float(), proposed)
        delta = proposed_error - current_error
        tolerance = (
            torch.finfo(torch.float32).eps
            * current_error.abs().clamp_min(1)
        )
        deltas.append(float(delta.item()))
        tolerances.append(float(tolerance.item()))
        safe &= bool((delta <= tolerance).item())
        strictly_improving |= bool((delta < -tolerance).item())
    accepted = safe and strictly_improving
    result = proposed if accepted else keep
    trace = {
        "proposed": True,
        "accepted": accepted,
        "outgoing_channel": outgoing,
        "incoming_channel": incoming,
        "constraint_deltas": deltas,
        "constraint_tolerances": tolerances,
    }
    return (result, trace) if return_trace else result


def guard_layer_swap_field(
    domain_grams: tuple[tuple[torch.Tensor, ...], ...],
    transition_grams: tuple[torch.Tensor, ...],
    base_keep_indices: tuple[torch.Tensor, ...],
    proposal_keep_indices: tuple[torch.Tensor, ...],
    *,
    return_trace: bool = False,
) -> tuple[torch.Tensor, ...] | tuple[
    tuple[torch.Tensor, ...], dict[str, object]
]:
    """Accept an entire cross-expert swap field under aggregate constraints.

    Each expert may contribute at most one same-width exchange.  The field is
    indivisible: every proposal is accepted only when every capability-domain
    reconstruction error is aggregate non-increasing and the aggregate
    trajectory-transition error strictly improves at FP32 scale.
    """

    expert_count = len(base_keep_indices)
    if expert_count == 0:
        raise ValueError("layer swap field must contain at least one expert")
    if len(proposal_keep_indices) != expert_count:
        raise ValueError("base and proposal fields must contain the same experts")
    if len(transition_grams) != expert_count:
        raise ValueError("transition grams must contain one gram per expert")
    if not domain_grams:
        raise ValueError("layer swap field requires capability-domain grams")
    if any(len(grams) != expert_count for grams in domain_grams):
        raise ValueError("every capability domain must contain one gram per expert")

    normalized_base: list[torch.Tensor] = []
    normalized_proposal: list[torch.Tensor] = []
    proposed_swaps = 0
    for expert_index, (raw_base, raw_proposal) in enumerate(
        zip(base_keep_indices, proposal_keep_indices)
    ):
        transition = transition_grams[expert_index]
        _validate_gram(transition)
        channel_count = transition.shape[0]

        def normalize_optional(raw: torch.Tensor, name: str) -> torch.Tensor:
            keep = torch.as_tensor(
                raw, dtype=torch.long, device=transition.device
            ).flatten()
            if keep.numel() == 0:
                return keep
            try:
                return _normalize_keep_indices(
                    keep, channel_count, transition.device
                )
            except ValueError as error:
                raise ValueError(f"{name} expert {expert_index}: {error}") from error

        base = normalize_optional(raw_base, "base")
        proposal = normalize_optional(raw_proposal, "proposal")
        if base.numel() != proposal.numel():
            raise ValueError("layer swap proposals must preserve expert widths")
        base_set = set(base.cpu().tolist())
        proposal_set = set(proposal.cpu().tolist())
        outgoing = len(base_set - proposal_set)
        incoming = len(proposal_set - base_set)
        if outgoing != incoming or outgoing > 1:
            raise ValueError("each expert may propose at most one same-width swap")
        proposed_swaps += outgoing
        normalized_base.append(base)
        normalized_proposal.append(proposal)

        for grams in domain_grams:
            gram = grams[expert_index]
            _validate_gram(gram)
            if gram.shape != transition.shape or gram.device != transition.device:
                raise ValueError(
                    "capability and transition grams must share shape and device"
                )

    def aggregate_delta(
        grams: tuple[torch.Tensor, ...]
    ) -> tuple[float, float]:
        base_error = 0.0
        proposal_error = 0.0
        for gram, base, proposal in zip(
            grams, normalized_base, normalized_proposal
        ):
            gram = gram.float()
            base_error += float(gram_reconstruction_error(gram, base).item())
            proposal_error += float(
                gram_reconstruction_error(gram, proposal).item()
            )
        tolerance = torch.finfo(torch.float32).eps * max(abs(base_error), 1.0)
        return proposal_error - base_error, tolerance

    domain_pairs = tuple(aggregate_delta(grams) for grams in domain_grams)
    transition_delta, transition_tolerance = aggregate_delta(transition_grams)
    accepted = (
        proposed_swaps > 0
        and all(delta <= tolerance for delta, tolerance in domain_pairs)
        and transition_delta < -transition_tolerance
    )
    selected = normalized_proposal if accepted else normalized_base
    result = tuple(keep.clone() for keep in selected)
    trace: dict[str, object] = {
        "proposed_swaps": proposed_swaps,
        "accepted": accepted,
        "domain_deltas": [delta for delta, _ in domain_pairs],
        "domain_tolerances": [tolerance for _, tolerance in domain_pairs],
        "transition_delta": transition_delta,
        "transition_tolerance": transition_tolerance,
    }
    return (result, trace) if return_trace else result


def pareto_swap_refine_keep_indices(
    grams: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    initial_keep_indices: torch.Tensor,
    *,
    max_swaps: int | None = None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, object]]]:
    """Refine a retained set using only three-view Pareto-safe 1-swaps.

    Candidate swaps must not increase any exact Gram reconstruction objective
    beyond its FP32-scale numerical tolerance and must strictly improve at
    least one objective.  Ties are resolved lexicographically by solution,
    full-trajectory, general-view delta, then channel indices.
    """

    if len(grams) != 3:
        raise ValueError("pareto refinement requires exactly three grams")
    if max_swaps is not None and max_swaps < 0:
        raise ValueError("max_swaps must be non-negative or None")
    for gram in grams:
        _validate_gram(gram)
    reference = grams[0]
    if any(gram.shape != reference.shape for gram in grams[1:]):
        raise ValueError("pareto grams must have the same shape")
    if any(gram.device != reference.device for gram in grams[1:]):
        raise ValueError("pareto grams must share a device")

    normalized_grams = tuple(
        gram.to(dtype=torch.float32) for gram in grams
    )
    raw_keep = torch.as_tensor(
        initial_keep_indices, dtype=torch.long, device=reference.device
    ).flatten()
    if raw_keep.numel() == 0:
        return (raw_keep, []) if return_trace else raw_keep
    keep = _normalize_keep_indices(raw_keep, reference.shape[0], reference.device)
    selected = torch.zeros(
        reference.shape[0], dtype=torch.bool, device=reference.device
    )
    selected[keep] = True
    if keep.numel() == reference.shape[0] or max_swaps == 0:
        result = torch.nonzero(selected, as_tuple=False).flatten()
        return (result, []) if return_trace else result

    visited = {tuple(torch.sort(keep).values.cpu().tolist())}
    trace: list[dict[str, object]] = []
    while max_swaps is None or len(trace) < max_swaps:
        retained = torch.nonzero(selected, as_tuple=False).flatten()
        pruned = torch.nonzero(~selected, as_tuple=False).flatten()
        pruned_indicator = (~selected).to(torch.float32)
        domain_deltas: list[torch.Tensor] = []
        tolerances: list[torch.Tensor] = []
        for gram in normalized_grams:
            residual_dot = gram @ pruned_indicator
            diagonal = gram.diag()
            delta = (
                2
                * (
                    residual_dot[retained, None]
                    - residual_dot[pruned][None, :]
                )
                + diagonal[retained, None]
                + diagonal[pruned][None, :]
                - 2 * gram[retained[:, None], pruned[None, :]]
            )
            current_error = pruned_indicator @ gram @ pruned_indicator
            tolerance = (
                torch.finfo(torch.float32).eps
                * current_error.abs().clamp_min(1)
            )
            domain_deltas.append(delta)
            tolerances.append(tolerance)

        deltas = torch.stack(domain_deltas)
        tolerance = torch.stack(tolerances)
        admissible = (deltas <= tolerance[:, None, None]).all(dim=0)
        admissible &= (deltas < -tolerance[:, None, None]).any(dim=0)
        if not admissible.any():
            break

        accepted = False
        while admissible.any():
            finalists = admissible.clone()
            for domain_index in (2, 1, 0):
                domain_values = deltas[domain_index].masked_fill(
                    ~finalists, float("inf")
                )
                minimum = domain_values.min()
                finalists &= deltas[domain_index] == minimum
            row, column = torch.nonzero(finalists, as_tuple=False)[0]
            outgoing = retained[row]
            incoming = pruned[column]
            next_selected = selected.clone()
            next_selected[outgoing] = False
            next_selected[incoming] = True
            key = tuple(
                torch.nonzero(next_selected, as_tuple=False)
                .flatten()
                .cpu()
                .tolist()
            )
            if key in visited:
                admissible[row, column] = False
                continue
            selected = next_selected
            visited.add(key)
            trace.append(
                {
                    "outgoing_channel": int(outgoing.item()),
                    "incoming_channel": int(incoming.item()),
                    "deltas": [
                        float(value)
                        for value in deltas[:, row, column].cpu().tolist()
                    ],
                    "tolerances": [
                        float(value) for value in tolerance.cpu().tolist()
                    ],
                }
            )
            accepted = True
            break
        if not accepted:
            break

    result = torch.nonzero(selected, as_tuple=False).flatten()
    return (result, trace) if return_trace else result


def _validate_swiglu_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> None:
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("SwiGLU weights must all be matrices")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have identical shapes")
    if down_weight.shape[1] != gate_weight.shape[0]:
        raise ValueError("down_weight input width must match the SwiGLU channel count")


def joint_mask_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    keep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cloned gate/up/down weights with all pruned channels zeroed."""

    _validate_swiglu_weights(gate_weight, up_weight, down_weight)
    keep = _normalize_keep_indices(
        keep_indices, gate_weight.shape[0], gate_weight.device
    )
    retained = torch.zeros(
        gate_weight.shape[0], dtype=torch.bool, device=gate_weight.device
    )
    retained[keep] = True
    masked_gate = gate_weight.clone()
    masked_up = up_weight.clone()
    masked_down = down_weight.clone()
    masked_gate[~retained] = 0
    masked_up[~retained] = 0
    masked_down[:, ~retained] = 0
    return masked_gate, masked_up, masked_down


def slice_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    keep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physically slice a SwiGLU expert to the retained channels."""

    _validate_swiglu_weights(gate_weight, up_weight, down_weight)
    keep = _normalize_keep_indices(
        keep_indices, gate_weight.shape[0], gate_weight.device
    )
    return gate_weight[keep], up_weight[keep], down_weight[:, keep]


def swiglu_output(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the bias-free SwiGLU expert used by Qwen3-MoE."""

    _validate_swiglu_weights(gate_weight, up_weight, down_weight)
    if inputs.shape[-1] != gate_weight.shape[1]:
        raise ValueError("input hidden size must match the projection weights")
    hidden = F.silu(inputs @ gate_weight.T) * (inputs @ up_weight.T)
    return hidden @ down_weight.T
