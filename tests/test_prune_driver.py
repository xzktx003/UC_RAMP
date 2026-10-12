import hashlib
import json
from types import SimpleNamespace

import torch
import pytest
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from scripts.prune_qwen3 import (
    allocate_calibration_samples,
    append_loop_contrast_calibration,
    append_loop_onset_calibration,
    allocate_balanced_categories,
    assistant_chat_end_predictor_token_mask,
    assistant_chat_response_predictor_token_mask,
    assistant_end_predictor_token_mask,
    assistant_solution_token_mask,
    capability_domain_batch_counts,
    specialist_domain_batch_counts,
    trajectory_phase_token_masks,
    trajectory_field_layer_enabled,
    capture_first_layer_inputs,
    capture_dense_loop_states,
    decoder_layer_hidden,
    domain_general_prefix_count,
    format_code_trace,
    format_reasoning_trace,
    layer_budget_prune_ratio,
    layer_capacity_aggregation,
    layer_statistics_token_mask,
    load_competition_math_exclusion_hashes,
    normalized_problem_sha256,
    layer_refinement_budget,
    next_output_metric_scale,
    pareto_domain_batch_counts,
    pareto_trust_region_swap_budget,
    parse_args,
    select_ramp_camera_mask_with_expert_widths,
    select_layer_mask,
)


def test_pareto_trust_region_reuses_the_one_eighth_locality_bound() -> None:
    assert pareto_trust_region_swap_budget(0) == 0
    assert pareto_trust_region_swap_budget(1) == 1
    assert pareto_trust_region_swap_budget(7) == 1
    assert pareto_trust_region_swap_budget(8) == 1


def test_trajectory_field_uses_only_the_fixed_final_third() -> None:
    assert [trajectory_field_layer_enabled(index, 48) for index in range(48)] == (
        [False] * 32 + [True] * 16
    )
    with pytest.raises(ValueError, match="divisible by 3"):
        trajectory_field_layer_enabled(0, 47)
    assert pareto_trust_region_swap_budget(16) == 2


def test_pareto_trust_region_rejects_a_negative_width() -> None:
    with pytest.raises(ValueError, match="retained_channels"):
        pareto_trust_region_swap_budget(-1)


def test_tail_protected_budget_is_exactly_global_fifty_percent() -> None:
    ratios = [layer_budget_prune_ratio("tail_protected", layer, 48) for layer in range(48)]
    explicit = [
        layer_budget_prune_ratio("tail_protected", layer, 48, 0.5)
        for layer in range(48)
    ]

    assert explicit == ratios
    assert ratios[:32] == [0.5625] * 32
    assert ratios[32:] == [0.375] * 16
    assert sum(ratios) / len(ratios) == 0.5
    assert layer_budget_prune_ratio("uniform", 17, 48) == 0.5


def test_tail_protected_budget_is_deterministic_exact_seventy_five_percent_keep() -> None:
    first = [
        layer_budget_prune_ratio("tail_protected", layer, 48, 0.25)
        for layer in range(48)
    ]
    second = [
        layer_budget_prune_ratio("tail_protected", layer, 48, 0.25)
        for layer in range(48)
    ]

    assert first == second
    assert first[:32] == [0.34375] * 32
    assert first[32:] == [0.0625] * 16
    assert sum(first) / len(first) == 0.25
    layer_capacity = 128 * 768
    retained = sum(round(layer_capacity * (1 - ratio)) for ratio in first)
    assert retained == 3_538_944
    assert retained * 4 == 3 * 4_718_592
    assert layer_budget_prune_ratio("uniform", 17, 48, 0.25) == 0.25


def test_tail_protected_budget_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_budget_prune_ratio("tail_protected", 0, 47)


def test_tail_difference_capacity_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_tail_difference_capacity", layer, 48
        )
        for layer in range(48)
    ]

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["anchored_difference_capacity"] * 16
    assert (
        layer_capacity_aggregation("anchored_geometric_capacity", 17, 48)
        == "anchored_geometric_capacity"
    )


def test_tail_difference_capacity_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation("anchored_tail_difference_capacity", 0, 47)


def test_tail_pareto_refinement_freezes_the_first_two_thirds() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "tail_pareto_residual_refinement", layer, 48
        )
        for layer in range(48)
    ]

    assert aggregations[:32] == ["pareto_prefix_frozen"] * 32
    assert aggregations[32:] == ["pareto_residual_refinement"] * 16


def test_tail_pareto_refinement_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation(
            "tail_pareto_residual_refinement", 0, 47
        )


def test_tail_code_complement_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_tail_code_complement_capacity", layer, 48
        )
        for layer in range(48)
    ]

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["capability_complement_capacity"] * 16


def test_tail_code_complement_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation(
            "anchored_tail_code_complement_capacity", 0, 47
        )


def test_dual_specialist_capacity_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "reasoning_early_tail_dual_specialist_capacity", layer, 48
        )
        for layer in range(48)
    ]

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["dual_specialist_capacity"] * 16
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation(
            "reasoning_early_tail_dual_specialist_capacity", 0, 47
        )


def test_tail_domain_robust_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_early_tail_domain_robust", layer, 48
        )
        for layer in range(48)
    ]

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["domain_robust"] * 16


def test_tail_domain_robust_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation("anchored_early_tail_domain_robust", 0, 47)


def test_solution_target_robust_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_early_tail_solution_domain_robust", layer, 48
        )
        for layer in range(48)
    ]
    statistics_mask = torch.tensor([[True, False, False]])

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["domain_robust"] * 16
    assert layer_statistics_token_mask(
        "anchored_early_tail_solution_domain_robust",
        31,
        48,
        statistics_mask,
    ).tolist() == [[True, True, True]]
    assert torch.equal(
        layer_statistics_token_mask(
            "anchored_early_tail_solution_domain_robust",
            32,
            48,
            statistics_mask,
        ),
        statistics_mask,
    )
    assert (
        layer_statistics_token_mask(
            "domain_robust", 17, 48, statistics_mask
        )
        is statistics_mask
    )


def test_solution_target_robust_rejects_incompatible_depth() -> None:
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_statistics_token_mask(
            "anchored_early_tail_solution_domain_robust",
            0,
            47,
            torch.ones((1, 1), dtype=torch.bool),
        )


def test_difficulty_termination_robust_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_early_tail_phase_domain_robust", layer, 48
        )
        for layer in range(48)
    ]
    statistics_mask = torch.tensor([[True, False, False]])

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["phase_domain_robust"] * 16
    assert layer_statistics_token_mask(
        "anchored_early_tail_phase_domain_robust",
        31,
        48,
        statistics_mask,
    ).all()
    assert torch.equal(
        layer_statistics_token_mask(
            "anchored_early_tail_phase_domain_robust",
            32,
            48,
            statistics_mask,
        ),
        statistics_mask,
    )
    with pytest.raises(ValueError, match="divisible by 3"):
        layer_capacity_aggregation(
            "anchored_early_tail_phase_domain_robust", 0, 47
        )


def test_phase_capacity_envelope_is_restricted_to_final_third() -> None:
    aggregations = [
        layer_capacity_aggregation(
            "anchored_early_tail_phase_capacity_envelope", layer, 48
        )
        for layer in range(48)
    ]
    statistics_mask = torch.tensor([[True, False, False]])

    assert aggregations[:32] == ["anchored_geometric_capacity"] * 32
    assert aggregations[32:] == ["phase_capacity_envelope"] * 16
    assert layer_statistics_token_mask(
        "anchored_early_tail_phase_capacity_envelope",
        31,
        48,
        statistics_mask,
    ).all()
    assert torch.equal(
        layer_statistics_token_mask(
            "anchored_early_tail_phase_capacity_envelope",
            32,
            48,
            statistics_mask,
        ),
        statistics_mask,
    )


def test_reasoning_mix_has_fixed_general_gsm_and_mathqa_allocation() -> None:
    assert allocate_calibration_samples(256, "reasoning_mix") == {
        "wikitext": 128,
        "gsm8k": 64,
        "math_qa": 64,
    }
    assert allocate_calibration_samples(256, "wikitext") == {
        "wikitext": 256
    }


def test_reasoning_code_mix_reserves_quarter_budget_for_mbpp_train() -> None:
    assert allocate_calibration_samples(256, "reasoning_code_mix") == {
        "wikitext": 128,
        "gsm8k": 32,
        "math_qa": 32,
        "mbpp": 64,
    }
    with pytest.raises(ValueError, match="divisible by 8"):
        allocate_calibration_samples(252, "reasoning_code_mix")


def test_reasoning_code_append_preserves_reasoning_prefix_before_mbpp() -> None:
    assert allocate_calibration_samples(320, "reasoning_code_append") == {
        "wikitext": 128,
        "gsm8k": 64,
        "math_qa": 64,
        "mbpp": 64,
    }
    with pytest.raises(ValueError, match="divisible by 5"):
        allocate_calibration_samples(321, "reasoning_code_append")


def test_competition_math_code_append_preserves_hardmath_prefix() -> None:
    assert allocate_calibration_samples(
        320, "competition_math_code_append"
    ) == {
        "wikitext": 128,
        "competition_math": 128,
        "mbpp": 64,
    }
    with pytest.raises(ValueError, match="divisible by 5"):
        allocate_calibration_samples(321, "competition_math_code_append")


def test_reasoning_hardmath_code_append_preserves_reasoning_prefix() -> None:
    assert allocate_calibration_samples(
        448, "reasoning_hardmath_code_append"
    ) == {
        "wikitext": 128,
        "gsm8k": 64,
        "math_qa": 64,
        "competition_math": 128,
        "mbpp": 64,
    }
    with pytest.raises(ValueError, match="divisible by 7"):
        allocate_calibration_samples(449, "reasoning_hardmath_code_append")


def test_pareto_append_allocation_preserves_solution_prefix() -> None:
    allocation = allocate_calibration_samples(
        384, "competition_math_solution_reasoning_append"
    )

    assert allocation == {
        "wikitext": 128,
        "competition_math": 128,
        "gsm8k": 64,
        "math_qa": 64,
    }
    with pytest.raises(ValueError, match="divisible by 6"):
        allocate_calibration_samples(
            385, "competition_math_solution_reasoning_append"
        )


def test_pareto_domain_batch_counts_split_append_boundaries() -> None:
    assert pareto_domain_batch_counts(124, 132, 128, 128) == (4, 4, 0)
    assert pareto_domain_batch_counts(252, 260, 128, 128) == (0, 4, 4)


def test_specialist_batch_counts_follow_append_only_boundaries() -> None:
    assert specialist_domain_batch_counts(126, 132, 128, 128, 128) == (
        2,
        4,
        0,
        0,
    )
    assert specialist_domain_batch_counts(254, 386, 128, 128, 128) == (
        0,
        2,
        128,
        2,
    )


def test_competition_math_mix_splits_general_and_hard_math_evenly() -> None:
    assert allocate_calibration_samples(256, "competition_math_mix") == {
        "wikitext": 128,
        "competition_math": 128,
    }
    with pytest.raises(ValueError, match="divisible by 4"):
        allocate_calibration_samples(255, "competition_math_mix")
    assert allocate_calibration_samples(
        256, "competition_math_solution_mix"
    ) == {
        "wikitext": 128,
        "competition_math": 128,
    }


def test_assistant_solution_token_mask_is_causally_shifted_and_includes_end() -> None:
    mask = assistant_solution_token_mask(
        torch.tensor([10, 1, 2, 11, 20, 10, 3, 4, 5, 11]),
        assistant_marker=torch.tensor([10, 3]),
        end_marker=torch.tensor([11]),
    )

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_assistant_end_predictor_mask_splits_solution_support_exactly() -> None:
    encoded = torch.tensor(
        [
            [10, 1, 2, 11, 20, 10, 3, 4, 5, 11],
            [10, 3, 6, 7, 11, 0, 0, 0, 0, 0],
        ]
    )
    solution_mask = torch.stack(
        [
            assistant_solution_token_mask(
                row,
                assistant_marker=torch.tensor([10, 3]),
                end_marker=torch.tensor([11]),
            )
            for row in encoded
        ]
    )
    end_mask = assistant_end_predictor_token_mask(
        encoded,
        end_marker=torch.tensor([11]),
        assistant_solution_mask=solution_mask,
    )
    continuation_mask = solution_mask & ~end_mask

    assert end_mask.tolist() == [
        [False, False, False, False, False, False, False, False, True, False],
        [False, False, False, True, False, False, False, False, False, False],
    ]
    assert not bool((continuation_mask & end_mask).any().item())
    assert torch.equal(continuation_mask | end_mask, solution_mask)


def test_assistant_chat_end_mask_ignores_user_and_incomplete_spans() -> None:
    encoded = torch.tensor(
        [
            [10, 1, 11, 20, 30, 3, 4, 11, 0, 0],
            [10, 1, 11, 20, 30, 5, 6, 7, 0, 0],
            [10, 1, 11, 20, 21, 5, 11, 0, 0, 0],
        ]
    )

    mask = assistant_chat_end_predictor_token_mask(
        encoded,
        assistant_marker=torch.tensor([20, 30]),
        end_marker=torch.tensor([11]),
    )

    assert mask.tolist() == [
        [False, False, False, False, False, False, True, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
    ]


def test_assistant_chat_response_mask_partitions_complete_continuation_and_end() -> None:
    encoded = torch.tensor(
        [
            [10, 1, 11, 20, 30, 3, 4, 11, 0, 0],
            [10, 1, 11, 20, 30, 5, 6, 7, 0, 0],
        ]
    )

    response = assistant_chat_response_predictor_token_mask(
        encoded,
        assistant_marker=torch.tensor([20, 30]),
        end_marker=torch.tensor([11]),
    )
    end = assistant_chat_end_predictor_token_mask(
        encoded,
        assistant_marker=torch.tensor([20, 30]),
        end_marker=torch.tensor([11]),
    )
    continuation = response & ~end

    assert response.tolist() == [
        [False, False, False, False, True, True, True, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
    ]
    assert end.sum().item() == 1
    assert continuation.sum().item() == 2
    assert not bool((continuation & end).any().item())


def test_trajectory_phase_masks_are_disjoint_and_cover_the_math_suffix() -> None:
    solution = torch.tensor(
        [[True, True, True, True], [False, True, True, False], [False, False, True, True]]
    )
    end = torch.tensor(
        [[False, False, False, False], [False, False, True, False], [False, False, False, True]]
    )

    problem, continuation, termination = trajectory_phase_token_masks(
        solution, end, general_sequence_count=1
    )

    assert not bool((problem & continuation).any())
    assert not bool((problem & termination).any())
    assert not bool((continuation & termination).any())
    assert not bool(problem[0].any())
    assert not bool(continuation[0].any())
    assert not bool(termination[0].any())
    assert bool((problem[1:] | continuation[1:] | termination[1:]).all())


def test_balanced_category_allocation_is_exact_and_nearly_uniform() -> None:
    allocation = allocate_balanced_categories(128, [
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus",
    ])

    assert sum(allocation.values()) == 128
    assert max(allocation.values()) - min(allocation.values()) == 1
    assert list(allocation) == sorted(allocation)


def test_competition_math_exclusion_manifest_filters_normalized_hashes(
    tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.write_bytes(b"fixed local train source")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "exclusions.json"
    expected_hash = normalized_problem_sha256("  A\N{FULLWIDTH PLUS SIGN}B  ")
    manifest.write_text(
        json.dumps(
            {
                "ngram_jaccard_threshold": 0.5,
                "competition_math_source_sha256": source_sha256,
                "excluded_normalized_problem_sha256": [expected_hash],
            }
        ),
        encoding="utf-8",
    )

    assert load_competition_math_exclusion_hashes(manifest, source) == {
        expected_hash
    }

    source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="source SHA256"):
        load_competition_math_exclusion_hashes(manifest, source)

    manifest.write_text(
        json.dumps(
            {
                "ngram_jaccard_threshold": 0.7,
                "excluded_normalized_problem_sha256": [expected_hash],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="threshold 0.5"):
        load_competition_math_exclusion_hashes(manifest)


def test_capability_domain_batch_counts_split_crossing_batches() -> None:
    assert capability_domain_batch_counts(0, 4, 128, 64) == (4, 0, 0)
    assert capability_domain_batch_counts(126, 132, 128, 64) == (2, 4, 0)
    assert capability_domain_batch_counts(190, 196, 128, 64) == (0, 2, 4)
    assert capability_domain_batch_counts(200, 204, 128, 64) == (0, 0, 4)


def test_capability_domain_batch_counts_reject_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="non-negative and ordered"):
        capability_domain_batch_counts(4, 3, 128, 64)


def test_reasoning_mix_requires_four_way_divisible_sample_budget() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        allocate_calibration_samples(126, "reasoning_mix")


def test_reasoning_trace_uses_chat_boundaries_and_step_by_step_prompt() -> None:
    trace = format_reasoning_trace("What is 2+2?", "2+2=4.")

    assert trace.startswith("<|im_start|>user\n")
    assert "Please reason step by step" in trace
    assert "<|im_start|>assistant\n2+2=4." in trace
    assert trace.endswith("<|im_end|>\n")


def test_code_trace_uses_train_solution_and_tests_without_humaneval() -> None:
    trace = format_code_trace(
        "Write a function that adds one.",
        "def add_one(x):\n    return x + 1",
        ["assert add_one(2) == 3"],
    )

    assert trace.startswith("<|im_start|>user\n")
    assert "assert add_one(2) == 3" in trace
    assert "<|im_start|>assistant\ndef add_one" in trace
    assert trace.endswith("<|im_end|>\n")


def test_anchored_robust_capacity_accepts_reasoning_code_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_code_mix",
            "--calibration-aggregation",
            "anchored_robust_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_corpus == "reasoning_code_mix"


def test_domain_robust_aggregation_requires_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-aggregation",
            "domain_robust",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_domain_anchored_aggregation_requires_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-aggregation",
            "domain_anchored",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_hierarchical_robust_aggregation_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "hierarchical_robust",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "hierarchical_robust"


def test_hierarchical_robust_aggregation_requires_ramp_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "hierarchical_robust",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_bounded_hierarchical_robust_aggregation_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "bounded_hierarchical_robust",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "bounded_hierarchical_robust"


def test_bounded_domain_anchored_aggregation_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "bounded_domain_anchored",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "bounded_domain_anchored"


def test_anchored_capacity_trust_region_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "anchored_capacity_trust_region",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "anchored_capacity_trust_region"


def test_anchored_robust_capacity_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "anchored_robust_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "anchored_robust_capacity"


def test_anchored_geometric_capacity_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "anchored_geometric_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "anchored_geometric_capacity"


def test_anchored_difference_capacity_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "anchored_difference_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "anchored_difference_capacity"


def test_anchored_tail_difference_capacity_accepts_reasoning_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_mix",
            "--calibration-aggregation",
            "anchored_tail_difference_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert (
        parse_args().calibration_aggregation
        == "anchored_tail_difference_capacity"
    )


def test_capability_envelope_accepts_only_reasoning_code_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_code_mix",
        "--calibration-aggregation",
        "capability_envelope_capacity",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert parse_args().calibration_aggregation == "capability_envelope_capacity"

    argv[argv.index("reasoning_code_mix")] = "reasoning_mix"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_capability_union_accepts_reasoning_code_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_code_mix",
            "--calibration-aggregation",
            "capability_union",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_aggregation == "capability_union"


def test_capability_complement_accepts_reasoning_code_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "reasoning_code_mix",
            "--calibration-aggregation",
            "capability_complement_capacity",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert (
        parse_args().calibration_aggregation
        == "capability_complement_capacity"
    )


def test_tail_code_complement_requires_reasoning_code_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_code_append",
        "--calibration-aggregation",
        "anchored_tail_code_complement_capacity",
        "--nsamples",
        "320",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert (
        parse_args().calibration_aggregation
        == "anchored_tail_code_complement_capacity"
    )

    argv[argv.index("reasoning_code_append")] = "reasoning_code_mix"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_tail_code_complement_accepts_hardmath_code_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--calibration-corpus",
            "competition_math_code_append",
            "--calibration-aggregation",
            "anchored_tail_code_complement_capacity",
            "--competition-math-source",
            str(source),
            "--competition-math-exclusion-manifest",
            str(exclusion_manifest),
            "--nsamples",
            "320",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().calibration_corpus == "competition_math_code_append"


def test_dual_specialist_capacity_accepts_only_append_only_four_domains(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "reasoning_early_tail_dual_specialist_capacity",
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()
    assert args.calibration_corpus == "reasoning_hardmath_code_append"
    assert (
        args.calibration_aggregation
        == "reasoning_early_tail_dual_specialist_capacity"
    )

    argv[argv.index("reasoning_hardmath_code_append")] = (
        "competition_math_code_append"
    )
    argv[argv.index("448")] = "320"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_universal_capability_envelope_accepts_only_append_only_four_domains(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "universal_capability_envelope",
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()
    assert args.calibration_corpus == "reasoning_hardmath_code_append"
    assert args.calibration_aggregation == "universal_capability_envelope"

    argv[argv.index("reasoning_hardmath_code_append")] = "reasoning_code_append"
    argv[argv.index("448")] = "320"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_universal_termination_envelope_accepts_only_append_only_four_domains(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "universal_termination_capacity_envelope",
        "--layer-budget-schedule",
        "tail_protected",
        "--pruning-rate",
        "0.25",
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()
    assert args.calibration_aggregation == "universal_termination_capacity_envelope"
    assert args.layer_budget_schedule == "tail_protected"
    assert args.prune_ratio == 0.25

    argv[argv.index("reasoning_hardmath_code_append")] = "reasoning_code_append"
    argv[argv.index("448")] = "320"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_pareto_residual_requires_paired_append_corpus_and_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "competition_math_solution_reasoning_append",
        "--calibration-aggregation",
        "pareto_residual_refinement",
        "--pareto-base-checkpoint",
        str(base),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "384",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.calibration_aggregation == "pareto_residual_refinement"
    assert args.pareto_base_checkpoint == base

    argv[argv.index("pareto_residual_refinement")] = (
        "tail_pareto_residual_refinement"
    )
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert args.calibration_aggregation == "tail_pareto_residual_refinement"

    argv[argv.index("competition_math_solution_reasoning_append")] = (
        "competition_math_solution_mix"
    )
    argv[argv.index("384")] = "256"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_trajectory_transition_refinement_requires_solution_corpus_and_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "competition_math_solution_mix",
        "--calibration-aggregation",
        "trajectory_transition_refinement",
        "--pareto-base-checkpoint",
        str(base),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "256",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.calibration_aggregation == "trajectory_transition_refinement"
    assert args.pareto_base_checkpoint == base

    argv[argv.index("trajectory_transition_refinement")] = (
        "trajectory_transition_guarded_refinement"
    )
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert (
        args.calibration_aggregation
        == "trajectory_transition_guarded_refinement"
    )

    argv[argv.index("competition_math_solution_mix")] = "competition_math_mix"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_tail_capability_transition_field_requires_universal_append_and_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "tail_capability_pareto_trajectory_field",
        "--pareto-base-checkpoint",
        str(base),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--max-swaps",
        "1",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert (
        args.calibration_aggregation
        == "tail_capability_pareto_trajectory_field"
    )
    assert args.pareto_base_checkpoint == base

    argv[argv.index("tail_capability_pareto_trajectory_field")] = (
        "tail_capability_trajectory_field"
    )
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert args.calibration_aggregation == "tail_capability_trajectory_field"

    argv[argv.index("reasoning_hardmath_code_append")] = "reasoning_code_mix"
    argv[argv.index("448")] = "256"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_router_neighborhood_refinement_requires_universal_append_base_and_swap1(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "router_neighborhood_local_separation_refinement",
        "--pareto-base-checkpoint",
        str(base),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--max-swaps",
        "1",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()
    assert (
        args.calibration_aggregation
        == "router_neighborhood_local_separation_refinement"
    )
    assert args.pareto_base_checkpoint == base

    argv[argv.index("1")] = "2"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_self_induced_loop_refinement_requires_manifest_base_and_swap1(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    loop_manifest = tmp_path / "loops.json"
    loop_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "self_induced_loop_orthogonal_refinement",
        "--pareto-base-checkpoint",
        str(base),
        "--loop-recovery-manifest",
        str(loop_manifest),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--max-swaps",
        "1",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.loop_recovery_manifest == loop_manifest
    argv[argv.index("1")] = "2"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_dense_loop_state_restoration_requires_manifest_base_and_swap1(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    loop_manifest = tmp_path / "loops.json"
    loop_manifest.write_text("{}", encoding="utf-8")
    base = tmp_path / "base.pt"
    base.touch()
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "reasoning_hardmath_code_append",
        "--calibration-aggregation",
        "causally_aligned_dense_loop_state_restoration",
        "--pareto-base-checkpoint",
        str(base),
        "--loop-recovery-manifest",
        str(loop_manifest),
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--nsamples",
        "448",
        "--max-swaps",
        "1",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    args = parse_args()

    assert args.loop_recovery_manifest == loop_manifest
    assert (
        args.calibration_aggregation
        == "causally_aligned_dense_loop_state_restoration"
    )
    argv[argv.index("1")] = "2"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


class _WhitespaceTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {"<|im_end|>": 99}

    def apply_chat_template(self, *args, **kwargs) -> list[int]:
        return [10, 11]

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        values = []
        for word in text.split():
            if word not in self._vocabulary:
                self._vocabulary[word] = len(self._vocabulary) + 100
            values.append(self._vocabulary[word])
        return SimpleNamespace(input_ids=values)


def test_loop_contrast_calibration_builds_causally_shifted_paired_branches(
    tmp_path,
) -> None:
    repeated_unit = "one two three four five six seven eight"
    output = " ".join([repeated_unit] * 4)
    from scripts.analyze_math_generation import (
        first_repeated_ngram_occurrence,
        repeated_ngram_fraction,
    )

    occurrence = first_repeated_ngram_occurrence(output, n=8)
    assert occurrence is not None
    manifest = {
        "schema_version": 1,
        "experiment_type": "train_only_self_induced_loop_mining",
        "protocol": {
            "temperature": 0,
            "causal_length": 2048,
            "ngram_size": 8,
            "repetition_threshold": 0.25,
            "seed": 1234,
            "test_outputs_read": False,
                "synthetic_loop_fallback": False,
                "reserved_assistant_end_tokens": 1,
        },
        "summary": {"samples": 1, "high_repetition": 1},
        "samples": [
            {
                "domain": "reasoning",
                "source": "gsm8k/train",
                "row_id": "7",
                "prompt": "question",
                "gold_response": "gold answer",
                "prompt_sha256": hashlib.sha256(b"question").hexdigest(),
                "gold_response_sha256": hashlib.sha256(
                    b"gold answer"
                ).hexdigest(),
                "model_output": output,
                "model_output_sha256": hashlib.sha256(
                    output.encode()
                ).hexdigest(),
                "repeated_ngram_fraction": repeated_ngram_fraction(output, n=8),
                "high_repetition": True,
                "first_repeated_ngram": occurrence,
            }
        ],
    }
    path = tmp_path / "loops.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    base = torch.arange(64, dtype=torch.long).unsqueeze(0)
    base_mask = torch.ones_like(base, dtype=torch.bool)

    samples, statistics, loop_mask, gold_mask, metadata = (
        append_loop_contrast_calibration(
            base,
            base_mask,
            _WhitespaceTokenizer(),
            manifest_path=path,
            sequence_length=64,
        )
    )

    assert samples.shape == statistics.shape == loop_mask.shape == gold_mask.shape
    assert samples.shape == (3, 64)
    assert not bool(statistics[1:].any())
    assert not bool(loop_mask[0].any()) and not bool(gold_mask[0].any())
    assert int(loop_mask[1].sum()) > 0 and not bool(loop_mask[2].any())
    assert int(gold_mask[2].sum()) == 3 and not bool(gold_mask[1].any())
    assert not bool((loop_mask & gold_mask).any())
    assert metadata["valid_pair_count"] == 1


def test_loop_onset_calibration_uses_one_identical_causal_predictor(
    tmp_path,
) -> None:
    repeated_unit = "one two three four five six seven eight"
    output = " ".join([repeated_unit] * 4)
    from scripts.analyze_math_generation import (
        first_repeated_ngram_occurrence,
        repeated_ngram_fraction,
    )

    occurrence = first_repeated_ngram_occurrence(output, n=8)
    assert occurrence is not None
    manifest = {
        "schema_version": 1,
        "experiment_type": "train_only_self_induced_loop_mining",
        "protocol": {
            "temperature": 0,
            "causal_length": 2048,
            "ngram_size": 8,
            "repetition_threshold": 0.25,
            "seed": 1234,
            "test_outputs_read": False,
            "synthetic_loop_fallback": False,
            "reserved_assistant_end_tokens": 1,
        },
        "summary": {"samples": 1, "high_repetition": 1},
        "samples": [
            {
                "domain": "code",
                "source": "mbpp/train",
                "row_id": "3",
                "prompt": "question",
                "gold_response": "gold answer",
                "prompt_sha256": hashlib.sha256(b"question").hexdigest(),
                "gold_response_sha256": hashlib.sha256(
                    b"gold answer"
                ).hexdigest(),
                "model_output": output,
                "model_output_sha256": hashlib.sha256(
                    output.encode()
                ).hexdigest(),
                "repeated_ngram_fraction": repeated_ngram_fraction(output, n=8),
                "high_repetition": True,
                "first_repeated_ngram": occurrence,
            }
        ],
    }
    path = tmp_path / "loops.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    base = torch.arange(64, dtype=torch.long).unsqueeze(0)
    base_mask = torch.ones_like(base, dtype=torch.bool)

    samples, statistics, onset_mask, metadata = append_loop_onset_calibration(
        base,
        base_mask,
        _WhitespaceTokenizer(),
        manifest_path=path,
        sequence_length=64,
    )

    assert samples.shape == statistics.shape == onset_mask.shape == (2, 64)
    assert not bool(statistics[1].any())
    assert not bool(onset_mask[0].any())
    assert int(onset_mask[1].sum()) == 1
    predictor = int(torch.nonzero(onset_mask[1]).item())
    assert predictor == metadata["contexts"][0]["predictor_index"]
    assert metadata["valid_context_count"] == 1
    assert metadata["contexts"][0]["domain"] == "code"


def test_competition_math_mix_requires_local_train_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    source = tmp_path / "train.parquet"
    source.touch()
    exclusion_manifest = tmp_path / "exclusions.json"
    exclusion_manifest.write_text("{}", encoding="utf-8")
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--calibration-corpus",
        "competition_math_mix",
        "--calibration-aggregation",
        "domain_robust",
        "--competition-math-source",
        str(source),
        "--competition-math-exclusion-manifest",
        str(exclusion_manifest),
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert args.competition_math_source == source
    assert args.competition_math_exclusion_manifest == exclusion_manifest

    argv[argv.index("domain_robust")] = "anchored_early_tail_domain_robust"
    monkeypatch.setattr("sys.argv", argv)
    assert (
        parse_args().calibration_aggregation
        == "anchored_early_tail_domain_robust"
    )

    argv[argv.index("anchored_early_tail_domain_robust")] = (
        "anchored_early_tail_solution_domain_robust"
    )
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()

    argv[argv.index("competition_math_mix")] = (
        "competition_math_solution_mix"
    )
    monkeypatch.setattr("sys.argv", argv)
    assert (
        parse_args().calibration_aggregation
        == "anchored_early_tail_solution_domain_robust"
    )

    assert parse_args().calibration_corpus == "competition_math_solution_mix"

    argv[argv.index("anchored_early_tail_solution_domain_robust")] = (
        "anchored_early_tail_phase_domain_robust"
    )
    monkeypatch.setattr("sys.argv", argv)
    assert (
        parse_args().calibration_aggregation
        == "anchored_early_tail_phase_domain_robust"
    )

    argv[argv.index("anchored_early_tail_phase_domain_robust")] = (
        "anchored_early_tail_phase_capacity_envelope"
    )
    monkeypatch.setattr("sys.argv", argv)
    assert (
        parse_args().calibration_aggregation
        == "anchored_early_tail_phase_capacity_envelope"
    )

    del argv[argv.index("--competition-math-source") : argv.index("--competition-math-source") + 2]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_domain_batch_split_handles_before_crossing_and_after_boundary() -> None:
    assert domain_general_prefix_count(0, 4, 8) == 4
    assert domain_general_prefix_count(6, 10, 8) == 2
    assert domain_general_prefix_count(8, 12, 8) == 0


def test_boundary_layer_refinement_budget_is_disabled_without_changing_others() -> None:
    skipped = {0, 47}

    assert layer_refinement_budget(1, 0, skipped) == 0
    assert layer_refinement_budget(1, 47, skipped) == 0
    assert layer_refinement_budget(1, 1, skipped) == 1


def test_current_transformers_decoder_slices_uneven_final_batch_kwargs() -> None:
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (5, 16))
    inputs, layer_kwargs = capture_first_layer_inputs(
        model,
        input_ids,
        device=torch.device("cpu"),
        chunk_size=4,
    )

    full_outputs = decoder_layer_hidden(
        model.model.layers[0],
        inputs[:4],
        layer_kwargs,
        captured_batch_size=4,
    )
    tail_outputs = decoder_layer_hidden(
        model.model.layers[0],
        inputs[4:],
        layer_kwargs,
        captured_batch_size=4,
    )

    assert full_outputs.shape == (4, 16, config.hidden_size)
    assert tail_outputs.shape == (1, 16, config.hidden_size)


def test_layer_replay_preserves_layer_specific_kwargs_and_shared_state() -> None:
    class SharedLayer(torch.nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

        def forward(
            self,
            hidden_states: torch.Tensor,
            positional_bias: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
            shared_kv_states: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            if self.index == 0:
                shared_kv_states["saved"] = hidden_states + attention_mask
                return hidden_states + positional_bias + attention_mask
            return hidden_states + positional_bias + shared_kv_states["saved"]

    class AlternatingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            self.model = SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=torch.nn.ModuleList(
                        [SharedLayer(0), SharedLayer(1)]
                    )
                )
            )

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.embedding

        def forward(
            self, input_ids: torch.Tensor, use_cache: bool = False
        ) -> torch.Tensor:
            del use_cache
            hidden_states = self.embedding(input_ids)
            shared: dict[str, torch.Tensor] = {}
            layers = self.model.language_model.layers
            for index, layer in enumerate(layers):
                positional_bias = hidden_states.new_full(
                    (hidden_states.shape[0], 1, 1), float(index + 1)
                )
                attention_mask = hidden_states.new_full(
                    (hidden_states.shape[0], 1, 1), float(10 * (index + 1))
                )
                hidden_states = layer(
                    hidden_states,
                    positional_bias,
                    attention_mask=attention_mask,
                    shared_kv_states=shared,
                )
            return hidden_states

    model = AlternatingModel().eval()
    input_ids = torch.randint(0, 32, (3, 5))
    inputs, replay = capture_first_layer_inputs(
        model,
        input_ids,
        device=torch.device("cpu"),
        chunk_size=2,
    )

    first = decoder_layer_hidden(
        model.model.language_model.layers[0],
        inputs[:2],
        replay,
        layer_index=0,
        batch_start=0,
        captured_batch_size=2,
    )
    second = decoder_layer_hidden(
        model.model.language_model.layers[1],
        first,
        replay,
        layer_index=1,
        batch_start=0,
        captured_batch_size=2,
    )

    torch.testing.assert_close(first, inputs[:2] + 11)
    torch.testing.assert_close(second, first + 2 + inputs[:2] + 10)

    tail_first = decoder_layer_hidden(
        model.model.language_model.layers[0],
        inputs[2:],
        replay,
        layer_index=0,
        batch_start=2,
        captured_batch_size=2,
    )
    tail_second = decoder_layer_hidden(
        model.model.language_model.layers[1],
        tail_first,
        replay,
        layer_index=1,
        batch_start=2,
        captured_batch_size=2,
    )
    assert tail_second.shape == (1, 5, 4)


def test_dense_loop_state_capture_preserves_layer_sample_and_expert_axes() -> None:
    class ToyExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 4, bias=False)
            self.up_proj = torch.nn.Linear(8, 4, bias=False)

    class ToyMoe(torch.nn.Module):
        top_k = 2
        norm_topk_prob = True

        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(8, 4, bias=False)
            self.experts = torch.nn.ModuleList([ToyExpert() for _ in range(4)])

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class ToyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMoe()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.mlp(hidden_states)

    class ToyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(32, 8)
            self.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])

        def forward(
            self, input_ids: torch.Tensor, *, use_cache: bool
        ) -> torch.Tensor:
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = ToyBackbone()
            self.config = SimpleNamespace(use_cache=True, hidden_size=8)

        def forward(
            self, input_ids: torch.Tensor, *, use_cache: bool
        ) -> torch.Tensor:
            return self.model(input_ids=input_ids, use_cache=use_cache)

    model = ToyModel().eval()
    input_ids = torch.randint(0, 32, (3, 16))
    selection = torch.zeros_like(input_ids, dtype=torch.bool)
    selection[0, 3] = True
    selection[1, 7] = True
    selection[2, 11] = True

    states, summary = capture_dense_loop_states(
        model,
        input_ids,
        selection,
        device=torch.device("cpu"),
        chunk_size=2,
    )

    assert len(states) == 2
    assert all(
        state.shape == (3, 4, 4)
        for state in states
    )
    assert all(torch.isfinite(state).all() for state in states)
    assert all(
        int((state.abs().sum(dim=2) > 0).sum(dim=1).max())
        <= 2
        for state in states
    )
    assert summary["samples"] == 3
    assert summary["selected_predictors"] == 3
    assert summary["chunk_size"] == 2


class _SyntheticAccumulator:
    channel_count = 4
    expert_count = 2
    device = torch.device("cpu")
    routed_token_counts = torch.tensor([10, 8])

    def camera_importance(self, _alpha: float) -> torch.Tensor:
        return torch.tensor(
            [[10.0, 9.0, 8.0, 1.0], [10.0, 9.0, 8.0, 1.0]]
        )

    def interaction_gram(self, _expert_index: int) -> torch.Tensor:
        return torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 2.5],
                [0.0, 0.0, 2.5, 1.0],
            ]
        )

    def reconstruction_gram(self, expert_index: int) -> torch.Tensor:
        return self.interaction_gram(expert_index)

    def reservoir_size_for(self, _expert_index: int) -> int:
        return 8


def test_global_ramp_preserves_camera_expert_width_allocation() -> None:
    accumulator = _SyntheticAccumulator()
    camera_mask, _, camera_coefficients = select_layer_mask(
        accumulator,
        method="camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
    )

    ramp_mask, records, ramp_coefficients = select_layer_mask(
        accumulator,
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
    )

    assert ramp_mask.sum().item() == camera_mask.sum().item() == 4
    torch.testing.assert_close(ramp_mask.sum(1), camera_mask.sum(1))
    assert len(records) == 2
    assert camera_coefficients is None
    assert ramp_coefficients is None


@pytest.mark.parametrize("method", ("camera", "diagonal", "ramp", "ramp_camera"))
def test_select_layer_mask_supports_exact_twenty_five_percent_pruning(
    method: str,
) -> None:
    mask, _, _ = select_layer_mask(
        _SyntheticAccumulator(),
        method=method,
        allocation="global" if method in {"camera", "ramp_camera"} else "uniform",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
        prune_ratio=0.25,
    )

    assert int(mask.sum().item()) == 6


def test_global_ramp_accepts_preallocated_expert_widths() -> None:
    mask, records, coefficients = select_ramp_camera_mask_with_expert_widths(
        _SyntheticAccumulator(),
        torch.tensor([1, 3]),
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
    )

    assert mask.sum(dim=1).tolist() == [1, 3]
    assert [record["retained_channels"] for record in records] == [1, 3]
    assert coefficients is None


def test_global_ramp_can_start_from_an_exact_frozen_stage1_mask() -> None:
    initial = torch.tensor(
        [[False, False, True, True], [True, False, False, False]]
    )
    mask, records, _ = select_ramp_camera_mask_with_expert_widths(
        _SyntheticAccumulator(),
        torch.tensor([2, 1]),
        camera_alpha=0.9,
        interaction_strength=0.2,
        max_swaps=0,
        initial_keep_mask=initial,
    )

    assert torch.equal(mask, initial)
    assert {record["initialization"] for record in records} == {
        "frozen_stage1_mask"
    }

    with pytest.raises(ValueError, match="match the fixed expert widths"):
        select_ramp_camera_mask_with_expert_widths(
            _SyntheticAccumulator(),
            torch.tensor([1, 1]),
            camera_alpha=0.9,
            interaction_strength=0.2,
            max_swaps=0,
            initial_keep_mask=initial,
        )


def test_global_ramp_can_decouple_allocation_and_refinement_importance() -> None:
    class _HierarchicalAccumulator(_SyntheticAccumulator):
        def allocation_importance(self, _alpha: float) -> torch.Tensor:
            return torch.tensor(
                [[100.0, 90.0, 80.0, 70.0], [4.0, 3.0, 2.0, 1.0]]
            )

    mask, _, _ = select_layer_mask(
        _HierarchicalAccumulator(),
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
    )

    assert mask.sum(dim=1).tolist() == [4, 0]


def test_global_ramp_records_capability_capacity_diagnostics() -> None:
    class _CapabilityAccumulator(_SyntheticAccumulator):
        def allocation_reference_importance(self, _alpha: float) -> torch.Tensor:
            return torch.tensor(
                [[10.0, 9.0, 8.0, 1.0], [10.0, 9.0, 8.0, 1.0]]
            )

        def allocation_importance_for_reference_widths(
            self, _alpha: float, _widths: torch.Tensor
        ) -> torch.Tensor:
            return torch.tensor(
                [[100.0, 90.0, 80.0, 70.0], [4.0, 3.0, 2.0, 1.0]]
            )

        def capacity_multipliers_for_reference_widths(
            self, _alpha: float, _widths: torch.Tensor
        ) -> torch.Tensor:
            return torch.tensor([2.0, 1.0])

    mask, records, _ = select_layer_mask(
        _CapabilityAccumulator(),
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=0,
    )

    assert mask.sum(dim=1).tolist() == [3, 1]
    assert [record["reference_retained_channels"] for record in records] == [2, 2]
    assert [record["capacity_multiplier"] for record in records] == [2.0, 1.0]
    assert [record["capacity_width_delta"] for record in records] == [1, -1]
    assert all("capacity_objective_improvement" in record for record in records)


def test_global_ramp_exposes_trust_radius_divisor_for_sensitivity() -> None:
    class _CapabilityAccumulator(_SyntheticAccumulator):
        def allocation_reference_importance(self, _alpha: float) -> torch.Tensor:
            return torch.tensor(
                [[10.0, 9.0, 8.0, 1.0], [10.0, 9.0, 8.0, 1.0]]
            )

        def allocation_importance_for_reference_widths(
            self, _alpha: float, _widths: torch.Tensor
        ) -> torch.Tensor:
            return torch.tensor(
                [[100.0, 90.0, 80.0, 70.0], [4.0, 3.0, 2.0, 1.0]]
            )

    mask, _, _ = select_layer_mask(
        _CapabilityAccumulator(),
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=0,
        max_width_delta_divisor=1,
    )
    assert mask.sum(dim=1).tolist() == [4, 0]

    with pytest.raises(ValueError, match="positive integer"):
        select_layer_mask(
            _CapabilityAccumulator(),
            method="ramp_camera",
            allocation="global",
            camera_alpha=0.9,
            interaction_strength=0.1,
            max_swaps=0,
            max_width_delta_divisor=0,
        )


def test_global_ramp_respects_camera_expert_floor() -> None:
    accumulator = _SyntheticAccumulator()

    ramp_mask, records, _ = select_layer_mask(
        accumulator,
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
        min_channels_per_expert=1,
    )

    assert ramp_mask.sum().item() == 4
    assert torch.all(ramp_mask.sum(dim=1) >= 1)
    retained = [record["retained_channels"] for record in records]
    assert sum(retained) == 4
    assert min(retained) >= 1


def test_global_ramp_can_recover_bounded_down_coefficients() -> None:
    mask, records, coefficients = select_layer_mask(
        _SyntheticAccumulator(),
        method="ramp_camera",
        allocation="global",
        camera_alpha=0.9,
        interaction_strength=0.1,
        max_swaps=1,
        coefficient_ridge_ratio=0.1,
        coefficient_clip_min=0.5,
        coefficient_clip_max=1.5,
    )

    assert coefficients is not None
    assert coefficients.shape == mask.shape
    assert torch.all((coefficients >= 0.5) & (coefficients <= 1.5))
    assert any(record["coefficient_min"] is not None for record in records)


def test_prune_driver_rejects_unregistered_pruning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "camera",
            "--prune-ratio",
            "0.4",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("pruning_rate", ("0.25", "0.5"))
def test_prune_driver_accepts_frozen_pruning_rate_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    pruning_rate: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "camera",
            "--pruning-rate",
            pruning_rate,
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().prune_ratio == float(pruning_rate)


def test_legacy_prune_ratio_flag_keeps_default_half_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "camera",
            "--prune-ratio",
            "0.5",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    assert parse_args().prune_ratio == 0.5


def test_prune_driver_accepts_centered_difference_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--interaction-metric",
            "centered_difference",
            "--difference-regularization",
            "0.01",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.interaction_metric == "centered_difference"
    assert args.difference_regularization == 0.01


def test_prune_driver_accepts_energy_calibrated_difference_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--interaction-metric",
            "energy_calibrated_difference",
            "--difference-strength",
            "1.0",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.interaction_metric == "energy_calibrated_difference"
    assert args.difference_strength == 1.0


def test_prune_driver_accepts_coefficient_difference_strength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--coefficient-ridge-ratio",
            "0.1",
            "--coefficient-difference-strength",
            "1.0",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.interaction_metric == "second_moment"
    assert args.coefficient_difference_strength == 1.0


def test_prune_driver_requires_coefficients_for_difference_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--coefficient-difference-strength",
            "1.0",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_prune_driver_accepts_deferred_coefficient_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "prune_qwen3.py",
        "--method",
        "ramp_camera",
        "--allocation",
        "global",
        "--coefficient-ridge-ratio",
        "0.1",
        "--defer-coefficient-application",
        "--mask-output",
        "mask.pt",
        "--result-output",
        "result.json",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert parse_args().defer_coefficient_application is True


def test_prune_driver_rejects_deferred_application_without_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prune_qwen3.py",
            "--method",
            "ramp_camera",
            "--allocation",
            "global",
            "--defer-coefficient-application",
            "--mask-output",
            "mask.pt",
            "--result-output",
            "result.json",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_next_rmsnorm_metric_uses_next_layer_and_final_norm() -> None:
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = Qwen3MoeForCausalLM(config)

    assert next_output_metric_scale(model, 0, "euclidean") is None
    assert (
        next_output_metric_scale(model, 0, "next_rmsnorm_gamma")
        is model.model.layers[1].input_layernorm.weight
    )
    assert (
        next_output_metric_scale(model, 1, "next_rmsnorm_gamma")
        is model.model.norm.weight
    )
    assert (
        next_output_metric_scale(model, 0, "next_rmsnorm_gamma_token_rms")
        is model.model.layers[1].input_layernorm.weight
    )
