import pytest
import torch

from ramp_moe.core import (
    apply_taylor_swap_prefix,
    build_centered_difference_interaction_gram,
    build_energy_calibrated_difference_interaction_gram,
    build_explicit_gram,
    build_interaction_gram,
    build_loop_orthogonal_importance,
    build_router_neighborhood_local_separation_gram,
    build_trajectory_transition_components,
    build_trajectory_transition_gram,
    diagonal_keep_indices,
    direct_reconstruction_error,
    forward_residual_keep_indices,
    gram_reconstruction_error,
    guard_layer_swap_field,
    guarded_swap_refine_keep_indices,
    halving_prefix_lengths,
    find_minimal_vector_feasible_pair,
    joint_mask_weights,
    minimum_norm_simplex_gradient,
    orthogonal_recovery_residuals,
    pareto_swap_refine_keep_indices,
    project_pareto_taylor_swap_prefix,
    reweight_gram_with_importance,
    recover_channel_coefficients,
    slice_weights,
    swap_refine_keep_indices,
    swiglu_output,
    universal_router_neighborhood_objective,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    routed_activations = torch.randn(11, 6, generator=generator)
    down_weight = torch.randn(5, 6, generator=generator)
    return routed_activations, down_weight


def test_factorized_gram_matches_explicit_contribution_gram() -> None:
    routed_activations, down_weight = _fixture()

    factorized = build_interaction_gram(routed_activations, down_weight)
    explicit = build_explicit_gram(routed_activations, down_weight)

    torch.testing.assert_close(factorized, explicit, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(factorized, factorized.T, rtol=0, atol=1e-6)
    assert torch.linalg.eigvalsh(factorized).min() >= -1e-5


def test_router_neighborhood_local_separation_uses_deterministic_nearest_neighbor() -> None:
    activations = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=torch.float32
    )
    signatures = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    down = torch.eye(2)
    ordinary = build_interaction_gram(activations, down)

    local, trace = build_router_neighborhood_local_separation_gram(
        activations, signatures, down, ordinary_gram=ordinary, return_trace=True
    )

    # Rows 0/1 select each other. Row 2 is tied between 0/1 and must select 0.
    assert trace["neighbor_indices"] == [1, 0, 0]
    assert trace["pair_count"] == 3
    assert trace["fallback"] is False
    torch.testing.assert_close(local.trace(), ordinary.trace())
    assert torch.linalg.eigvalsh(local).min() >= -1e-5


def test_router_neighborhood_local_separation_falls_back_when_degenerate() -> None:
    activations = torch.tensor([[2.0, 1.0]])
    signatures = torch.tensor([[0.5, 0.5]])
    ordinary = torch.diag(torch.tensor([3.0, 1.0]))

    local, trace = build_router_neighborhood_local_separation_gram(
        activations,
        signatures,
        torch.eye(2),
        ordinary_gram=ordinary,
        return_trace=True,
    )

    torch.testing.assert_close(local, ordinary)
    assert trace["pair_count"] == 0
    assert trace["fallback"] is True


def test_universal_router_neighborhood_objective_uses_fixed_four_domain_anchor() -> None:
    ordinary = tuple(
        torch.diag(torch.tensor([float(index + 1), 1.0]))
        for index in range(4)
    )
    local = tuple(torch.flip(gram, dims=(0, 1)) for gram in ordinary)

    objective = universal_router_neighborhood_objective(ordinary, local)

    specialist_weight = 0.1 / 3
    anchor = 0.9 * ordinary[0] + specialist_weight * sum(ordinary[1:])
    local_anchor = 0.9 * local[0] + specialist_weight * sum(local[1:])
    matched = local_anchor * (anchor.trace() / local_anchor.trace())
    torch.testing.assert_close(objective, 0.5 * (anchor + matched))
    torch.testing.assert_close(objective.trace(), anchor.trace())


def test_loop_orthogonal_importance_removes_the_loop_parallel_component() -> None:
    loop = torch.tensor([[1.0, 0.0]])
    gold = torch.tensor([[2.0, 3.0]])
    anchor = torch.ones(2)

    importance, trace = build_loop_orthogonal_importance(
        loop,
        gold,
        torch.eye(2),
        anchor_importance=anchor,
        return_trace=True,
    )

    assert trace["fallback"] is False
    torch.testing.assert_close(
        orthogonal_recovery_residuals(loop, gold, torch.eye(2)),
        torch.tensor([[0.0, 3.0]]),
    )
    assert importance[1] > importance[0]


def test_loop_orthogonal_importance_falls_back_for_parallel_gold() -> None:
    anchor = torch.tensor([3.0, 1.0])

    importance, trace = build_loop_orthogonal_importance(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[2.0, 0.0]]),
        torch.eye(2),
        anchor_importance=anchor,
        return_trace=True,
    )

    torch.testing.assert_close(importance, anchor)
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "zero_orthogonal_recovery"


def test_loop_orthogonal_importance_preserves_tiny_nonzero_recovery() -> None:
    anchor = torch.ones(2)

    importance, trace = build_loop_orthogonal_importance(
        torch.tensor([[1.0e-4, 0.0]]),
        torch.tensor([[1.0e-4, 2.0e-4]]),
        torch.eye(2),
        anchor_importance=anchor,
        return_trace=True,
    )

    assert trace["fallback"] is False
    assert trace["recovery_mass_before_match"] > 0
    assert torch.isfinite(importance).all()
    torch.testing.assert_close(importance.square().sum(), anchor.sum())


def test_trajectory_transition_gram_trace_matches_each_observed_phase() -> None:
    anchor = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    down_weight = torch.tensor([[1.0, 2.0, 0.5], [0.5, -1.0, 2.0]])
    problem = torch.tensor([1.0, 0.0, 0.0])
    solution = torch.tensor([0.0, 2.0, 0.0])
    termination = torch.tensor([0.0, 0.0, 3.0])

    actual = build_trajectory_transition_gram(
        anchor,
        down_weight,
        problem_mean=problem,
        solution_mean=solution,
        termination_mean=termination,
    )

    weight_gram = down_weight.T @ down_weight
    pd = torch.outer(solution - problem, solution - problem) * weight_gram
    dt = torch.outer(termination - solution, termination - solution) * weight_gram
    expected = (
        anchor
        + pd * (anchor.trace() / pd.trace())
        + dt * (anchor.trace() / dt.trace())
    ) / 3
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.trace(), anchor.trace())
    assert torch.linalg.eigvalsh(actual).min() >= -1e-5


def test_trajectory_transition_gram_skips_unobserved_or_degenerate_phases() -> None:
    anchor = torch.diag(torch.tensor([2.0, 1.0]))
    down_weight = torch.eye(2)
    same = torch.tensor([1.0, 1.0])

    actual = build_trajectory_transition_gram(
        anchor,
        down_weight,
        problem_mean=same,
        solution_mean=same,
        termination_mean=None,
    )

    torch.testing.assert_close(actual, anchor)


def test_trajectory_transition_components_keep_observed_phases_separate() -> None:
    anchor = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    down_weight = torch.tensor([[1.0, 2.0, 0.5], [0.5, -1.0, 2.0]])
    problem = torch.tensor([1.0, 0.0, 0.0])
    solution = torch.tensor([0.0, 2.0, 0.0])
    termination = torch.tensor([0.0, 0.0, 3.0])

    absolute, problem_to_solution, solution_to_termination = (
        build_trajectory_transition_components(
            anchor,
            down_weight,
            problem_mean=problem,
            solution_mean=solution,
            termination_mean=termination,
        )
    )

    assert problem_to_solution is not None
    assert solution_to_termination is not None
    torch.testing.assert_close(absolute, anchor)
    torch.testing.assert_close(problem_to_solution.trace(), anchor.trace())
    torch.testing.assert_close(solution_to_termination.trace(), anchor.trace())
    torch.testing.assert_close(
        build_trajectory_transition_gram(
            anchor,
            down_weight,
            problem_mean=problem,
            solution_mean=solution,
            termination_mean=termination,
        ),
        torch.stack(
            (absolute, problem_to_solution, solution_to_termination)
        ).mean(dim=0),
    )


def test_centered_difference_gram_trace_matches_output_contribution_energy() -> None:
    routed_activations, down_weight = _fixture()
    regularization = 0.01

    actual = build_centered_difference_interaction_gram(
        routed_activations,
        down_weight,
        regularization=regularization,
    )

    ordinary = build_interaction_gram(routed_activations, down_weight)
    centered = routed_activations - routed_activations.mean(dim=0, keepdim=True)
    difference = build_interaction_gram(centered, down_weight)
    trace_ratio = ordinary.trace() / difference.trace()
    expected = regularization * ordinary
    expected += (1 - regularization) * trace_ratio * difference

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.trace(), ordinary.trace())


def test_all_pair_differences_equal_twice_centered_contribution_gram() -> None:
    routed_activations, down_weight = _fixture()
    centered = routed_activations - routed_activations.mean(dim=0, keepdim=True)
    pair_differences = (
        routed_activations[:, None, :] - routed_activations[None, :, :]
    ).reshape(-1, routed_activations.shape[1])

    pair_gram = build_interaction_gram(pair_differences, down_weight)
    centered_gram = build_interaction_gram(centered, down_weight)

    torch.testing.assert_close(pair_gram, 2 * centered_gram)


def test_centered_difference_gram_falls_back_for_constant_activations() -> None:
    activations = torch.ones(4, 3)
    down_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    actual = build_centered_difference_interaction_gram(
        activations, down_weight, regularization=0.01
    )

    torch.testing.assert_close(actual, build_interaction_gram(activations, down_weight))


def test_centered_difference_gram_validates_regularization() -> None:
    routed_activations, down_weight = _fixture()

    for regularization in (-0.01, 1.01):
        with pytest.raises(ValueError, match="regularization"):
            build_centered_difference_interaction_gram(
                routed_activations,
                down_weight,
                regularization=regularization,
            )


def test_energy_calibrated_difference_adds_natural_centered_energy() -> None:
    routed_activations, down_weight = _fixture()
    strength = 1.0

    actual = build_energy_calibrated_difference_interaction_gram(
        routed_activations, down_weight, strength=strength
    )

    ordinary = build_interaction_gram(routed_activations, down_weight)
    centered = routed_activations - routed_activations.mean(dim=0, keepdim=True)
    expected = ordinary + strength * build_interaction_gram(centered, down_weight)
    torch.testing.assert_close(actual, expected)


def test_energy_calibrated_difference_zero_strength_is_ordinary() -> None:
    routed_activations, down_weight = _fixture()

    actual = build_energy_calibrated_difference_interaction_gram(
        routed_activations, down_weight, strength=0
    )

    torch.testing.assert_close(actual, build_interaction_gram(routed_activations, down_weight))


def test_energy_calibrated_difference_rejects_negative_strength() -> None:
    routed_activations, down_weight = _fixture()

    with pytest.raises(ValueError, match="strength"):
        build_energy_calibrated_difference_interaction_gram(
            routed_activations, down_weight, strength=-0.01
        )


def test_gram_objective_matches_direct_output_mse() -> None:
    routed_activations, down_weight = _fixture()
    gram = build_interaction_gram(routed_activations, down_weight)
    keep = torch.tensor([0, 2, 5])

    gram_error = gram_reconstruction_error(gram, keep)
    direct_error = direct_reconstruction_error(
        routed_activations, down_weight, keep
    )

    torch.testing.assert_close(gram_error, direct_error, rtol=1e-5, atol=1e-6)


def test_diagonal_solver_keeps_largest_diagonal_entries() -> None:
    gram = torch.diag(torch.tensor([1.0, 7.0, 2.0, 5.0]))

    keep = diagonal_keep_indices(gram, keep_count=2)

    assert keep.tolist() == [1, 3]
    assert gram_reconstruction_error(gram, keep).item() == 3.0


def test_forward_residual_step_uses_exact_marginal_gain() -> None:
    gram = torch.tensor(
        [
            [4.5972, 5.9252, 0.9658, -0.4789, -2.1858],
            [5.9252, 16.5363, 3.9446, -3.7967, -9.5791],
            [0.9658, 3.9446, 6.0269, -2.4553, -4.2302],
            [-0.4789, -3.7967, -2.4553, 3.5568, 4.2939],
            [-2.1858, -9.5791, -4.2302, 4.2939, 13.1531],
        ]
    )

    keep = forward_residual_keep_indices(gram, keep_count=2)

    # Static diagonal ranking would retain channels 1 and 4. Exact residual
    # gains instead select channel 0 followed by the complementary channel 2.
    assert keep.tolist() == [0, 2]

    one_keep_error = gram_reconstruction_error(gram, keep[:1])
    two_keep_error = gram_reconstruction_error(gram, keep)
    expected_gain = one_keep_error - two_keep_error
    residual_dot = gram.sum(dim=1) - gram[:, keep[0]]
    formula_gain = 2 * residual_dot[keep[1]] - gram[keep[1], keep[1]]
    torch.testing.assert_close(expected_gain, formula_gain)


def test_joint_mask_and_physical_slice_are_functionally_equivalent() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(4, 5, generator=generator)
    gate_weight = torch.randn(6, 5, generator=generator)
    up_weight = torch.randn(6, 5, generator=generator)
    down_weight = torch.randn(5, 6, generator=generator)
    keep = torch.tensor([0, 2, 5])

    masked = joint_mask_weights(gate_weight, up_weight, down_weight, keep)
    sliced = slice_weights(gate_weight, up_weight, down_weight, keep)

    masked_output = swiglu_output(x, *masked)
    sliced_output = swiglu_output(x, *sliced)
    torch.testing.assert_close(masked_output, sliced_output, rtol=1e-5, atol=1e-6)


def test_invalid_keep_count_is_rejected() -> None:
    gram = torch.eye(3)

    for keep_count in (0, 4):
        try:
            forward_residual_keep_indices(gram, keep_count)
        except ValueError as error:
            assert "keep_count" in str(error)
        else:
            raise AssertionError("invalid keep_count must raise ValueError")


def test_reweighted_gram_preserves_robust_importance_diagonal() -> None:
    gram = torch.tensor([[4.0, 2.0], [2.0, 9.0]])
    importance = torch.tensor([16.0, 25.0])

    reweighted = reweight_gram_with_importance(
        gram, importance, interaction_strength=0.1
    )

    torch.testing.assert_close(reweighted.diag(), importance)
    torch.testing.assert_close(reweighted[0, 1], torch.tensor(2.0 / 3.0))
    assert torch.linalg.eigvalsh(reweighted).min() >= -1e-6


def test_swap_refinement_preserves_budget_and_reduces_error() -> None:
    gram = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 2.5],
            [0.0, 0.0, 2.5, 1.0],
        ]
    )
    initial = torch.tensor([0, 1])

    refined = swap_refine_keep_indices(gram, initial, max_swaps=1)

    assert refined.numel() == initial.numel()
    assert gram_reconstruction_error(gram, refined) < gram_reconstruction_error(
        gram, initial
    )


def test_guarded_swap_accepts_only_the_fused_proposal_when_all_views_are_safe() -> None:
    proposal = torch.diag(torch.tensor([1.0, 5.0, 4.0]))
    constraints = (
        torch.diag(torch.tensor([1.0, 3.0, 2.0])),
        torch.diag(torch.tensor([2.0, 4.0, 3.0])),
    )
    initial = torch.tensor([0])

    refined, trace = guarded_swap_refine_keep_indices(
        proposal, constraints, initial, return_trace=True
    )

    assert refined.tolist() == [1]
    assert trace["accepted"] is True
    assert trace["outgoing_channel"] == 0
    assert trace["incoming_channel"] == 1
    assert all(delta < 0 for delta in trace["constraint_deltas"])


def test_guarded_swap_rejects_an_unsafe_best_proposal_without_searching_an_alternative() -> None:
    proposal = torch.diag(torch.tensor([1.0, 5.0, 4.0]))
    constraints = (
        torch.diag(torch.tensor([3.0, 1.0, 4.0])),
        torch.diag(torch.tensor([2.0, 1.0, 3.0])),
    )
    initial = torch.tensor([0])

    refined, trace = guarded_swap_refine_keep_indices(
        proposal, constraints, initial, return_trace=True
    )

    assert refined.tolist() == [0]
    assert trace["accepted"] is False
    assert trace["incoming_channel"] == 1
    assert trace["constraint_deltas"][0] > 0


def test_layer_swap_field_accepts_only_the_complete_aggregate_safe_field() -> None:
    base = (torch.tensor([0]), torch.tensor([0]))
    proposal = (torch.tensor([1]), torch.tensor([1]))
    domain_grams = (
        (
            torch.diag(torch.tensor([2.0, 1.0])),
            torch.diag(torch.tensor([1.0, 3.0])),
        ),
        (
            torch.diag(torch.tensor([1.0, 3.0])),
            torch.diag(torch.tensor([3.0, 1.0])),
        ),
    )
    transition_grams = (
        torch.diag(torch.tensor([1.0, 4.0])),
        torch.diag(torch.tensor([1.0, 3.0])),
    )

    selected, trace = guard_layer_swap_field(
        domain_grams,
        transition_grams,
        base,
        proposal,
        return_trace=True,
    )

    assert trace["accepted"] is True
    assert trace["proposed_swaps"] == 2
    assert trace["domain_deltas"][0] < 0
    assert trace["domain_deltas"][1] == 0
    assert trace["transition_delta"] < 0
    assert all(torch.equal(actual, expected) for actual, expected in zip(selected, proposal))


def test_layer_swap_field_rejects_every_proposal_when_one_domain_is_unsafe() -> None:
    base = (torch.tensor([0]), torch.empty(0, dtype=torch.long))
    proposal = (torch.tensor([1]), torch.empty(0, dtype=torch.long))
    domain_grams = (
        (
            torch.diag(torch.tensor([3.0, 1.0])),
            torch.eye(2),
        ),
    )
    transition_grams = (
        torch.diag(torch.tensor([1.0, 4.0])),
        torch.eye(2),
    )

    selected, trace = guard_layer_swap_field(
        domain_grams,
        transition_grams,
        base,
        proposal,
        return_trace=True,
    )

    assert trace["accepted"] is False
    assert trace["domain_deltas"][0] > trace["domain_tolerances"][0]
    assert trace["transition_delta"] < 0
    assert all(torch.equal(actual, expected) for actual, expected in zip(selected, base))


def test_pareto_swap_accepts_only_jointly_nonincreasing_exchange() -> None:
    grams = (
        torch.diag(torch.tensor([1.0, 3.0, 0.5])),
        torch.diag(torch.tensor([1.0, 2.0, 2.0])),
        torch.diag(torch.tensor([1.0, 4.0, 10.0])),
    )
    initial = torch.tensor([0])

    refined = pareto_swap_refine_keep_indices(grams, initial)

    assert refined.tolist() == [1]
    for gram in grams:
        assert gram_reconstruction_error(gram, refined) <= gram_reconstruction_error(
            gram, initial
        )


def test_pareto_swap_rejects_solution_gain_that_harms_general_view() -> None:
    grams = (
        torch.diag(torch.tensor([2.0, 1.0])),
        torch.diag(torch.tensor([1.0, 3.0])),
        torch.diag(torch.tensor([1.0, 4.0])),
    )
    initial = torch.tensor([0])

    refined = pareto_swap_refine_keep_indices(grams, initial)

    assert refined.tolist() == [0]


def test_pareto_swap_validates_exactly_three_matching_grams() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        pareto_swap_refine_keep_indices((torch.eye(2),), torch.tensor([0]))
    with pytest.raises(ValueError, match="same shape"):
        pareto_swap_refine_keep_indices(
            (torch.eye(2), torch.eye(3), torch.eye(2)), torch.tensor([0])
        )


def test_pareto_swap_preserves_an_empty_fixed_width_expert() -> None:
    grams = (torch.eye(3), torch.eye(3), torch.eye(3))

    refined, trace = pareto_swap_refine_keep_indices(
        grams, torch.empty(0, dtype=torch.long), return_trace=True
    )

    assert refined.numel() == 0
    assert trace == []


def test_gram_reconstruction_error_supports_an_empty_keep_set() -> None:
    gram = torch.tensor([[2.0, 0.5], [0.5, 3.0]])

    error = gram_reconstruction_error(
        gram, torch.empty(0, dtype=torch.long)
    )

    assert error.item() == pytest.approx(6.0)


def test_pareto_swap_respects_a_fixed_trust_region_budget() -> None:
    diagonal = torch.arange(1.0, 17.0)
    grams = tuple(torch.diag(diagonal * scale) for scale in (1.0, 1.1, 1.2))
    initial = torch.arange(8)

    refined, trace = pareto_swap_refine_keep_indices(
        grams, initial, max_swaps=1, return_trace=True
    )

    assert len(trace) == 1
    assert torch.isin(refined, initial).sum().item() == 7


def test_pareto_swap_rejects_a_negative_trust_region_budget() -> None:
    grams = (torch.eye(2), torch.eye(2), torch.eye(2))

    with pytest.raises(ValueError, match="max_swaps"):
        pareto_swap_refine_keep_indices(
            grams, torch.tensor([0]), max_swaps=-1
        )


def test_coefficient_recovery_solves_regularized_normal_equation() -> None:
    gram = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    keep = torch.tensor([0])

    coefficients = recover_channel_coefficients(
        gram,
        keep,
        ridge_ratio=0.1,
        clip_min=0.0,
        clip_max=2.0,
    )

    torch.testing.assert_close(coefficients, torch.tensor([1.6 / 1.1]))


def test_minimum_norm_simplex_gradient_is_scale_invariant() -> None:
    gradients = torch.tensor([[100.0, 0.0], [0.0, 1.0]])

    weights, combined = minimum_norm_simplex_gradient(gradients)

    torch.testing.assert_close(weights, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(combined, torch.tensor([0.5, 0.5]))


def test_minimum_norm_simplex_gradient_cancels_opposed_domains() -> None:
    gradients = torch.tensor([[2.0, 0.0], [-3.0, 0.0]])

    weights, combined = minimum_norm_simplex_gradient(gradients)

    torch.testing.assert_close(weights, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(combined, torch.zeros(2), atol=1e-6, rtol=0)


def test_pareto_taylor_prefix_accepts_cross_expert_compensation() -> None:
    # Both experts retain channel 0.  The first proposal improves the loop
    # domain while hurting the general domain; the second compensates the
    # general loss.  Neither proposal is independently Pareto-safe, but the
    # coordinated two-expert prefix is.
    domain_gradients = torch.zeros(2, 1, 2, 2)
    domain_gradients[:, 0, 0, 1] = torch.tensor([1.0, -3.0])
    domain_gradients[:, 0, 1, 1] = torch.tensor([-2.0, 0.0])
    base_mask = torch.tensor([[[True, False], [True, False]]])

    refined, trace = project_pareto_taylor_swap_prefix(
        domain_gradients,
        base_mask,
        combination_weights=torch.tensor([0.5, 0.5]),
        loop_domain=1,
        return_trace=True,
    )

    assert refined.tolist() == [[[False, True], [False, True]]]
    assert trace["selected_swaps"] == 2
    assert trace["domain_deltas"][0] == pytest.approx(-1 / 5**0.5)
    assert trace["domain_deltas"][1] == pytest.approx(-1.0)


def test_pareto_taylor_prefix_returns_base_when_no_joint_prefix_is_safe() -> None:
    domain_gradients = torch.zeros(2, 1, 1, 2)
    domain_gradients[:, 0, 0, 1] = torch.tensor([1.0, -3.0])
    base_mask = torch.tensor([[[True, False]]])

    refined, trace = project_pareto_taylor_swap_prefix(
        domain_gradients,
        base_mask,
        combination_weights=torch.tensor([0.5, 0.5]),
        loop_domain=1,
        return_trace=True,
    )

    assert torch.equal(refined, base_mask)
    assert trace["selected_swaps"] == 0
    assert trace["status"] == "no_pareto_prefix"


def test_taylor_swap_prefix_replays_an_exact_same_width_prefix() -> None:
    base = torch.tensor([[[True, False, True], [False, True, False]]])
    selected = [
        {"layer": 0, "expert": 0, "outgoing": 0, "incoming": 1},
        {"layer": 0, "expert": 1, "outgoing": 1, "incoming": 2},
    ]

    first = apply_taylor_swap_prefix(base, selected, 1)
    both = apply_taylor_swap_prefix(base, selected, 2)

    assert first.tolist() == [[[False, True, True], [False, True, False]]]
    assert both.tolist() == [[[False, True, True], [False, False, True]]]
    assert torch.equal(first.sum(dim=-1), base.sum(dim=-1))
    assert torch.equal(both.sum(dim=-1), base.sum(dim=-1))


def test_taylor_swap_prefix_rejects_a_record_that_does_not_match_base() -> None:
    base = torch.tensor([[[True, False]]])
    selected = [{"layer": 0, "expert": 0, "outgoing": 1, "incoming": 0}]

    with pytest.raises(ValueError, match="base mask"):
        apply_taylor_swap_prefix(base, selected, 1)


def test_halving_prefix_schedule_is_deterministic_and_reaches_one() -> None:
    assert halving_prefix_lengths(9) == [9, 4, 2, 1]
    assert halving_prefix_lengths(1) == [1]
    assert halving_prefix_lengths(0) == []


def test_vector_feasible_pair_finds_compensation_outside_every_prefix() -> None:
    # In the frozen order A, B, C, every non-empty prefix hurts the clean
    # domain.  The non-prefix pair A+C is jointly clean-safe and strictly
    # improves the counterexample domain.
    deltas = torch.tensor(
        [
            [2.0, -3.0],
            [2.0, 2.0],
            [-3.0, 0.0],
        ]
    )

    selected, trace = find_minimal_vector_feasible_pair(
        deltas,
        group_ids=torch.tensor([0, 1, 2]),
        clean_domains=(0,),
        strict_domains=(1,),
        return_trace=True,
    )

    assert selected.tolist() == [0, 2]
    assert trace["status"] == "selected_pair"
    assert trace["cardinality"] == 2
    assert trace["domain_deltas"] == pytest.approx([-1.0, -3.0])


def test_vector_feasible_pair_prefers_a_feasible_singleton() -> None:
    deltas = torch.tensor([[-1.0, -2.0], [-3.0, 1.0]])

    selected = find_minimal_vector_feasible_pair(
        deltas,
        group_ids=torch.tensor([0, 1]),
        clean_domains=(0,),
        strict_domains=(1,),
    )

    assert selected.tolist() == [0]


def test_vector_feasible_pair_rejects_same_group_and_returns_null() -> None:
    deltas = torch.tensor([[2.0, -3.0], [-3.0, 0.0]])

    selected, trace = find_minimal_vector_feasible_pair(
        deltas,
        group_ids=torch.tensor([7, 7]),
        clean_domains=(0,),
        strict_domains=(1,),
        return_trace=True,
    )

    assert selected.numel() == 0
    assert trace["status"] == "no_feasible_pair"
