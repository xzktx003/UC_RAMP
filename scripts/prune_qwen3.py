#!/usr/bin/env python3
"""Generate sequential fixed-endpoint Qwen3-MoE pruning masks."""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import unicodedata

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ramp_moe.core import (
    build_loop_orthogonal_importance,
    build_trajectory_transition_components,
    diagonal_keep_indices,
    forward_residual_keep_indices,
    gram_reconstruction_error,
    guard_layer_swap_field,
    guarded_swap_refine_keep_indices,
    pareto_swap_refine_keep_indices,
    reweight_gram_with_importance,
    recover_channel_coefficients,
    swap_refine_keep_indices,
    universal_router_neighborhood_objective,
)
from ramp_moe.evaluation import dataset_download_config
from ramp_moe.generation import (
    first_repeated_ngram_occurrence,
    repeated_ngram_fraction,
)
from ramp_moe.moe_adapter import resolve_decoder_layers, resolve_input_embeddings
from ramp_moe.qwen3_stats import (
    CapabilityComplementStatsAccumulator,
    CapabilityEnvelopeStatsAccumulator,
    CapabilityUnionStatsAccumulator,
    DomainRobustStatsAccumulator,
    DualSpecialistCapacityEnvelopeStatsAccumulator,
    MultiDomainRobustStatsAccumulator,
    PhaseCapacityEnvelopeStatsAccumulator,
    Qwen3LayerStatsAccumulator,
    router_weighted_phase_means,
    UniversalCapabilityEnvelopeStatsAccumulator,
    UniversalTerminationCapacityEnvelopeStatsAccumulator,
)
from ramp_moe.selection import (
    apply_qwen3_channel_coefficients_,
    apply_qwen3_keep_mask_,
    select_bounded_camera_keep_mask,
    select_camera_keep_mask,
    select_keep_mask_with_expert_widths,
)


MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
PRUNE_RATIO = 0.5
CAMERA_REVISION = "b67d084e29ea44060f03079096005608f3eb03a3"
COMPETITION_MATH_CORPORA = frozenset(
    {
        "competition_math_mix",
        "competition_math_solution_mix",
        "competition_math_code_append",
        "reasoning_hardmath_code_append",
        "competition_math_solution_reasoning_append",
    }
)


class _CaptureComplete(Exception):
    pass


@dataclass(frozen=True)
class _CapturedLayerCall:
    args: tuple[object, ...]
    kwargs: dict[str, object]


@dataclass
class LayerReplayContext:
    """Layer-specific captured arguments plus per-batch mutable replay state."""

    calls: tuple[_CapturedLayerCall, ...]
    _mutable_states: dict[tuple[str, int, str], object] = dataclass_field(
        default_factory=dict
    )

    def prepare(
        self,
        layer_index: int,
        *,
        batch_size: int,
        captured_batch_size: int,
        batch_start: int,
        replay_pass: str,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        if not 0 <= layer_index < len(self.calls):
            raise IndexError("layer_index is outside captured replay calls")
        call = self.calls[layer_index]
        args = tuple(
            _slice_captured_batch_value(
                value,
                batch_size=batch_size,
                captured_batch_size=captured_batch_size,
            )
            for value in call.args
        )
        kwargs: dict[str, object] = {}
        for name, value in call.kwargs.items():
            if isinstance(value, MutableMapping):
                state_key = (replay_pass, batch_start, name)
                if state_key not in self._mutable_states:
                    try:
                        self._mutable_states[state_key] = type(value)()
                    except TypeError:
                        self._mutable_states[state_key] = {}
                kwargs[name] = self._mutable_states[state_key]
            else:
                kwargs[name] = _slice_captured_batch_value(
                    value,
                    batch_size=batch_size,
                    captured_batch_size=captured_batch_size,
                )
        return args, kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("camera", "diagonal", "ramp", "ramp_camera"),
        required=True,
    )
    parser.add_argument(
        "--allocation", choices=("global", "uniform"), default="uniform"
    )
    parser.add_argument(
        "--pruning-rate",
        "--prune-ratio",
        dest="prune_ratio",
        type=float,
        choices=(0.25, PRUNE_RATIO),
        default=PRUNE_RATIO,
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-source")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--calibration-corpus",
        choices=(
            "wikitext",
            "reasoning_mix",
            "reasoning_code_mix",
            "reasoning_code_append",
            "competition_math_mix",
            "competition_math_solution_mix",
            "competition_math_code_append",
            "reasoning_hardmath_code_append",
            "competition_math_solution_reasoning_append",
        ),
        default="wikitext",
    )
    parser.add_argument("--competition-math-source", type=Path)
    parser.add_argument("--competition-math-exclusion-manifest", type=Path)
    parser.add_argument(
        "--calibration-aggregation",
        choices=(
            "pooled",
            "domain_robust",
            "domain_anchored",
            "bounded_domain_anchored",
            "anchored_capacity_trust_region",
            "anchored_robust_capacity",
            "anchored_geometric_capacity",
            "anchored_difference_capacity",
            "anchored_tail_difference_capacity",
            "anchored_early_tail_domain_robust",
            "anchored_early_tail_solution_domain_robust",
            "anchored_early_tail_phase_domain_robust",
            "anchored_early_tail_phase_capacity_envelope",
            "reasoning_early_tail_dual_specialist_capacity",
            "pareto_residual_refinement",
            "tail_pareto_residual_refinement",
            "trajectory_transition_refinement",
            "trajectory_transition_guarded_refinement",
            "capability_envelope_capacity",
            "universal_capability_envelope",
            "universal_termination_capacity_envelope",
            "tail_capability_pareto_trajectory_field",
            "tail_capability_trajectory_field",
            "router_neighborhood_local_separation_refinement",
            "self_induced_loop_orthogonal_refinement",
            "causally_aligned_dense_loop_state_restoration",
            "capability_complement_capacity",
            "anchored_tail_code_complement_capacity",
            "capability_union",
            "hierarchical_robust",
            "bounded_hierarchical_robust",
        ),
        default="pooled",
    )
    parser.add_argument(
        "--layer-budget-schedule",
        choices=("uniform", "tail_protected"),
        default="uniform",
    )
    parser.add_argument("--calib-length", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--tokens-per-expert", type=int, default=512)
    parser.add_argument("--camera-alpha", type=float, default=0.9)
    parser.add_argument("--interaction-strength", type=float, default=0.1)
    parser.add_argument(
        "--interaction-metric",
        choices=(
            "second_moment",
            "centered_difference",
            "energy_calibrated_difference",
        ),
        default="second_moment",
    )
    parser.add_argument("--difference-regularization", type=float, default=0.01)
    parser.add_argument("--difference-strength", type=float, default=1.0)
    parser.add_argument("--max-swaps", type=int, default=8)
    parser.add_argument(
        "--skip-refine-layer", type=int, action="append", default=[]
    )
    parser.add_argument("--coefficient-ridge-ratio", type=float)
    parser.add_argument(
        "--defer-coefficient-application", action="store_true"
    )
    parser.add_argument(
        "--coefficient-difference-strength", type=float, default=0.0
    )
    parser.add_argument("--coefficient-clip-min", type=float, default=0.5)
    parser.add_argument("--coefficient-clip-max", type=float, default=1.5)
    parser.add_argument("--min-channels-per-expert", type=int, default=0)
    parser.add_argument(
        "--output-metric",
        choices=(
            "euclidean",
            "next_rmsnorm_gamma",
            "next_rmsnorm_gamma_token_rms",
        ),
        default="euclidean",
    )
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--pareto-base-checkpoint", type=Path)
    parser.add_argument("--loop-recovery-manifest", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.method in {"diagonal", "ramp"} and args.allocation != "uniform":
        parser.error("diagonal and ramp require --allocation uniform")
    if args.method == "ramp_camera" and args.allocation != "global":
        parser.error("ramp_camera requires --allocation global")
    if args.layer_budget_schedule != "uniform" and (
        args.method != "ramp_camera" or args.allocation != "global"
    ):
        parser.error("non-uniform layer budgets require global ramp_camera")
    if args.nsamples <= 0 or args.calib_length <= 1 or args.chunk_size <= 0:
        parser.error("sample, sequence, and chunk sizes must be positive")
    if args.calibration_corpus == "reasoning_mix" and args.nsamples % 4:
        parser.error("reasoning_mix --nsamples must be divisible by 4")
    if args.calibration_corpus == "reasoning_code_mix" and args.nsamples % 8:
        parser.error("reasoning_code_mix --nsamples must be divisible by 8")
    if args.calibration_corpus == "reasoning_code_append" and args.nsamples % 5:
        parser.error("reasoning_code_append --nsamples must be divisible by 5")
    if (
        args.calibration_corpus == "competition_math_code_append"
        and args.nsamples % 5
    ):
        parser.error(
            "competition_math_code_append --nsamples must be divisible by 5"
        )
    if (
        args.calibration_corpus == "reasoning_hardmath_code_append"
        and args.nsamples % 7
    ):
        parser.error(
            "reasoning_hardmath_code_append --nsamples must be divisible by 7"
        )
    if (
        args.calibration_corpus
        == "competition_math_solution_reasoning_append"
        and args.nsamples % 6
    ):
        parser.error(
            "competition_math_solution_reasoning_append --nsamples must be divisible by 6"
        )
    if (
        args.calibration_corpus in COMPETITION_MATH_CORPORA
        and args.calibration_corpus != "competition_math_code_append"
        and args.calibration_corpus != "reasoning_hardmath_code_append"
        and args.calibration_corpus
        != "competition_math_solution_reasoning_append"
        and args.nsamples % 4
    ):
        parser.error("competition MATH calibration --nsamples must be divisible by 4")
    if args.calibration_corpus in COMPETITION_MATH_CORPORA:
        if (
            args.competition_math_source is None
            or not args.competition_math_source.is_file()
            or args.competition_math_source.suffix != ".parquet"
        ):
            parser.error(
                "competition MATH calibration requires an existing local train parquet"
            )
        if (
            args.competition_math_exclusion_manifest is None
            or not args.competition_math_exclusion_manifest.is_file()
            or args.competition_math_exclusion_manifest.suffix != ".json"
        ):
            parser.error(
                "competition MATH calibration requires a local JSON exclusion manifest"
            )
    elif args.competition_math_source is not None:
        parser.error(
            "--competition-math-source requires a competition MATH corpus"
        )
    elif args.competition_math_exclusion_manifest is not None:
        parser.error(
            "--competition-math-exclusion-manifest requires "
            "a competition MATH corpus"
        )
    if (
        args.calibration_aggregation != "pooled"
        and args.calibration_corpus
        not in {
            "reasoning_mix",
            "reasoning_code_mix",
            "reasoning_code_append",
            "competition_math_mix",
            "competition_math_solution_mix",
            "competition_math_code_append",
            "reasoning_hardmath_code_append",
            "competition_math_solution_reasoning_append",
        }
    ):
        parser.error("domain aggregation requires mixed-domain calibration")
    if (
        args.calibration_aggregation
        in {
            "capability_envelope_capacity",
            "capability_complement_capacity",
            "anchored_tail_code_complement_capacity",
            "capability_union",
        }
        and args.calibration_corpus
        not in {
            "reasoning_code_mix",
            "reasoning_code_append",
            "competition_math_code_append",
        }
    ):
        parser.error(
            "capability-domain aggregation requires a three-domain code corpus"
        )
    if (
        args.calibration_aggregation
        == "reasoning_early_tail_dual_specialist_capacity"
        and args.calibration_corpus != "reasoning_hardmath_code_append"
    ):
        parser.error(
            "dual-specialist capacity requires reasoning_hardmath_code_append"
        )
    if (
        args.calibration_aggregation
        in {
            "universal_capability_envelope",
            "universal_termination_capacity_envelope",
            "tail_capability_pareto_trajectory_field",
            "tail_capability_trajectory_field",
            "router_neighborhood_local_separation_refinement",
            "self_induced_loop_orthogonal_refinement",
            "causally_aligned_dense_loop_state_restoration",
        }
        and args.calibration_corpus != "reasoning_hardmath_code_append"
    ):
        parser.error(
            "universal capability envelope requires reasoning_hardmath_code_append"
        )
    if (
        args.calibration_aggregation
        == "anchored_tail_code_complement_capacity"
        and args.calibration_corpus
        not in {"reasoning_code_append", "competition_math_code_append"}
    ):
        parser.error(
            "anchored_tail_code_complement_capacity requires an append-only code corpus"
        )
    if (
        args.calibration_aggregation
        in {
            "anchored_early_tail_solution_domain_robust",
            "anchored_early_tail_phase_domain_robust",
            "anchored_early_tail_phase_capacity_envelope",
            "trajectory_transition_refinement",
            "trajectory_transition_guarded_refinement",
        }
        and args.calibration_corpus != "competition_math_solution_mix"
    ):
        parser.error(
            "solution/phase tail aggregation requires "
            "competition_math_solution_mix"
        )
    if (
        args.calibration_aggregation
        in {"pareto_residual_refinement", "tail_pareto_residual_refinement"}
        and args.calibration_corpus
        != "competition_math_solution_reasoning_append"
    ):
        parser.error(
            "pareto residual refinement requires its append-only paired corpus"
        )
    if args.calibration_aggregation in {
        "pareto_residual_refinement",
        "tail_pareto_residual_refinement",
        "trajectory_transition_refinement",
        "trajectory_transition_guarded_refinement",
        "tail_capability_pareto_trajectory_field",
        "tail_capability_trajectory_field",
        "router_neighborhood_local_separation_refinement",
        "self_induced_loop_orthogonal_refinement",
        "causally_aligned_dense_loop_state_restoration",
    }:
        if (
            args.pareto_base_checkpoint is None
            or not args.pareto_base_checkpoint.is_file()
        ):
            parser.error(
                "mask refinement requires an existing base checkpoint"
            )
    elif args.pareto_base_checkpoint is not None:
        parser.error(
            "--pareto-base-checkpoint requires a mask refinement"
        )
    if args.calibration_aggregation in {
        "self_induced_loop_orthogonal_refinement",
        "causally_aligned_dense_loop_state_restoration",
    }:
        if (
            args.loop_recovery_manifest is None
            or not args.loop_recovery_manifest.is_file()
        ):
            parser.error(
                "self-induced loop refinement requires a local loop manifest"
            )
    elif args.loop_recovery_manifest is not None:
        parser.error(
            "--loop-recovery-manifest requires self-induced loop refinement"
        )
    if (
        args.calibration_aggregation in {
            "bounded_domain_anchored",
            "anchored_capacity_trust_region",
            "anchored_robust_capacity",
            "anchored_geometric_capacity",
            "anchored_difference_capacity",
            "anchored_tail_difference_capacity",
            "anchored_early_tail_domain_robust",
            "anchored_early_tail_solution_domain_robust",
            "anchored_early_tail_phase_domain_robust",
            "anchored_early_tail_phase_capacity_envelope",
            "reasoning_early_tail_dual_specialist_capacity",
            "pareto_residual_refinement",
            "tail_pareto_residual_refinement",
            "trajectory_transition_refinement",
            "trajectory_transition_guarded_refinement",
            "capability_envelope_capacity",
            "universal_capability_envelope",
            "universal_termination_capacity_envelope",
            "tail_capability_pareto_trajectory_field",
            "tail_capability_trajectory_field",
            "router_neighborhood_local_separation_refinement",
            "self_induced_loop_orthogonal_refinement",
            "causally_aligned_dense_loop_state_restoration",
            "capability_complement_capacity",
            "anchored_tail_code_complement_capacity",
            "capability_union",
            "hierarchical_robust",
            "bounded_hierarchical_robust",
        }
        and args.method != "ramp_camera"
    ):
        parser.error("hierarchical robust aggregation requires ramp_camera")
    if args.nsamples % args.chunk_size:
        parser.error("--nsamples must be divisible by --chunk-size")
    if args.calib_length != 2048:
        parser.error("formal pruning calibration requires --calib-length 2048")
    if not 0 <= args.interaction_strength <= 1:
        parser.error("--interaction-strength must be in [0, 1]")
    if not 0 <= args.difference_regularization <= 1:
        parser.error("--difference-regularization must be in [0, 1]")
    if args.difference_strength < 0:
        parser.error("--difference-strength must be non-negative")
    if (
        args.interaction_metric != "second_moment"
        and args.method == "camera"
    ):
        parser.error("difference interaction metrics require a RAMP method")
    if args.max_swaps < 0:
        parser.error("--max-swaps must be non-negative")
    if (
        args.calibration_aggregation
        in {
            "tail_capability_pareto_trajectory_field",
            "tail_capability_trajectory_field",
            "router_neighborhood_local_separation_refinement",
            "self_induced_loop_orthogonal_refinement",
            "causally_aligned_dense_loop_state_restoration",
        }
        and args.max_swaps != 1
    ):
        parser.error("single-swap mask refinement requires --max-swaps 1")
    if args.skip_refine_layer and args.method != "ramp_camera":
        parser.error("--skip-refine-layer requires --method ramp_camera")
    if any(layer < 0 for layer in args.skip_refine_layer):
        parser.error("--skip-refine-layer values must be non-negative")
    args.skip_refine_layer = sorted(set(args.skip_refine_layer))
    if args.coefficient_ridge_ratio is not None:
        if args.method != "ramp_camera":
            parser.error("--coefficient-ridge-ratio requires --method ramp_camera")
        if args.coefficient_ridge_ratio <= 0:
            parser.error("--coefficient-ridge-ratio must be positive")
        if not 0 <= args.coefficient_clip_min <= args.coefficient_clip_max:
            parser.error("coefficient clip bounds must satisfy 0 <= min <= max")
    if args.coefficient_difference_strength < 0:
        parser.error("--coefficient-difference-strength must be non-negative")
    if (
        args.coefficient_difference_strength > 0
        and args.coefficient_ridge_ratio is None
    ):
        parser.error(
            "--coefficient-difference-strength requires --coefficient-ridge-ratio"
        )
    if (
        args.defer_coefficient_application
        and args.coefficient_ridge_ratio is None
    ):
        parser.error(
            "--defer-coefficient-application requires --coefficient-ridge-ratio"
        )
    if args.min_channels_per_expert < 0:
        parser.error("--min-channels-per-expert must be non-negative")
    if args.min_channels_per_expert and (
        args.allocation != "global"
        or args.method not in {"camera", "ramp_camera"}
    ):
        parser.error(
            "--min-channels-per-expert requires global camera or ramp_camera"
        )
    return args


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "datasets", "accelerate")
    return {name: importlib.metadata.version(name) for name in names}


def allocate_calibration_samples(
    nsamples: int, calibration_corpus: str
) -> dict[str, int]:
    """Return the fixed source allocation for a calibration protocol."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    if calibration_corpus == "wikitext":
        return {"wikitext": nsamples}
    if calibration_corpus == "reasoning_mix":
        if nsamples % 4:
            raise ValueError("reasoning_mix nsamples must be divisible by 4")
        quarter = nsamples // 4
        return {
            "wikitext": 2 * quarter,
            "gsm8k": quarter,
            "math_qa": quarter,
        }
    if calibration_corpus == "reasoning_code_mix":
        if nsamples % 8:
            raise ValueError("reasoning_code_mix nsamples must be divisible by 8")
        eighth = nsamples // 8
        return {
            "wikitext": 4 * eighth,
            "gsm8k": eighth,
            "math_qa": eighth,
            "mbpp": 2 * eighth,
        }
    if calibration_corpus == "reasoning_code_append":
        if nsamples % 5:
            raise ValueError(
                "reasoning_code_append nsamples must be divisible by 5"
            )
        fifth = nsamples // 5
        return {
            "wikitext": 2 * fifth,
            "gsm8k": fifth,
            "math_qa": fifth,
            "mbpp": fifth,
        }
    if calibration_corpus == "competition_math_code_append":
        if nsamples % 5:
            raise ValueError(
                "competition_math_code_append nsamples must be divisible by 5"
            )
        fifth = nsamples // 5
        return {
            "wikitext": 2 * fifth,
            "competition_math": 2 * fifth,
            "mbpp": fifth,
        }
    if calibration_corpus == "reasoning_hardmath_code_append":
        if nsamples % 7:
            raise ValueError(
                "reasoning_hardmath_code_append nsamples must be divisible by 7"
            )
        seventh = nsamples // 7
        return {
            "wikitext": 2 * seventh,
            "gsm8k": seventh,
            "math_qa": seventh,
            "competition_math": 2 * seventh,
            "mbpp": seventh,
        }
    if calibration_corpus == "competition_math_solution_reasoning_append":
        if nsamples % 6:
            raise ValueError(
                "competition_math_solution_reasoning_append nsamples must be divisible by 6"
            )
        sixth = nsamples // 6
        return {
            "wikitext": 2 * sixth,
            "competition_math": 2 * sixth,
            "gsm8k": sixth,
            "math_qa": sixth,
        }
    if calibration_corpus in COMPETITION_MATH_CORPORA:
        if nsamples % 4:
            raise ValueError(
                "competition MATH calibration nsamples must be divisible by 4"
            )
        half = nsamples // 2
        return {"wikitext": half, "competition_math": half}
    raise ValueError(f"unsupported calibration corpus: {calibration_corpus}")


def allocate_balanced_categories(
    sample_count: int, categories: list[str]
) -> dict[str, int]:
    """Split an exact sample budget almost uniformly over stable categories."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    normalized = sorted({str(category).strip() for category in categories})
    if not normalized or any(not category for category in normalized):
        raise ValueError("categories must contain non-empty unique names")
    base, remainder = divmod(sample_count, len(normalized))
    if base == 0:
        raise ValueError("sample_count must cover every category")
    return {
        category: base + int(index < remainder)
        for index, category in enumerate(normalized)
    }


def normalized_problem_sha256(problem: object) -> str:
    """Hash a problem after the fixed leakage-audit normalization."""

    normalized = " ".join(
        unicodedata.normalize("NFKC", str(problem)).lower().split()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_competition_math_exclusion_hashes(
    manifest: Path, competition_math_source: Path | None = None
) -> set[str]:
    """Load the preregistered MATH500 near-duplicate exclusions."""

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("ngram_jaccard_threshold") != 0.5:
        raise ValueError("competition MATH exclusion manifest must use threshold 0.5")
    if competition_math_source is not None and payload.get(
        "competition_math_source_sha256"
    ) != _sha256_file(competition_math_source):
        raise ValueError("competition MATH exclusion manifest source SHA256 mismatch")
    values = payload.get("excluded_normalized_problem_sha256")
    if not isinstance(values, list) or not values:
        raise ValueError("competition MATH exclusion manifest has no hashes")
    hashes = {str(value).lower() for value in values}
    if len(hashes) != len(values) or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        raise ValueError("competition MATH exclusion hashes must be unique SHA256 hex")
    return hashes


def domain_general_prefix_count(
    batch_start: int, batch_stop: int, general_sequence_count: int
) -> int:
    """Return how many leading sequences in a batch belong to the general domain."""

    if batch_start < 0 or batch_stop < batch_start or general_sequence_count < 0:
        raise ValueError("domain batch boundaries must be non-negative and ordered")
    return min(max(general_sequence_count - batch_start, 0), batch_stop - batch_start)


def capability_domain_batch_counts(
    batch_start: int,
    batch_stop: int,
    general_sequence_count: int,
    math_sequence_count: int,
) -> tuple[int, int, int]:
    """Split an ordered calibration batch into general, math, and code counts."""

    if (
        batch_start < 0
        or batch_stop < batch_start
        or general_sequence_count < 0
        or math_sequence_count < 0
    ):
        raise ValueError("domain batch boundaries must be non-negative and ordered")
    general_count = domain_general_prefix_count(
        batch_start, batch_stop, general_sequence_count
    )
    noncode_count = domain_general_prefix_count(
        batch_start,
        batch_stop,
        general_sequence_count + math_sequence_count,
    )
    math_count = noncode_count - general_count
    code_count = batch_stop - batch_start - noncode_count
    return general_count, math_count, code_count


def specialist_domain_batch_counts(
    batch_start: int,
    batch_stop: int,
    general_sequence_count: int,
    reasoning_sequence_count: int,
    hard_math_sequence_count: int,
) -> tuple[int, int, int, int]:
    """Split an append-only batch into general, reasoning, hard-math, and code."""

    if any(
        value < 0
        for value in (
            batch_start,
            batch_stop,
            general_sequence_count,
            reasoning_sequence_count,
            hard_math_sequence_count,
        )
    ) or batch_stop < batch_start:
        raise ValueError("specialist domain boundaries must be non-negative and ordered")
    boundaries = (
        general_sequence_count,
        general_sequence_count + reasoning_sequence_count,
        general_sequence_count
        + reasoning_sequence_count
        + hard_math_sequence_count,
    )
    cumulative = [
        domain_general_prefix_count(batch_start, batch_stop, boundary)
        for boundary in boundaries
    ]
    general_count, non_specialist_count, non_code_count = cumulative
    return (
        general_count,
        non_specialist_count - general_count,
        non_code_count - non_specialist_count,
        batch_stop - batch_start - non_code_count,
    )


def pareto_domain_batch_counts(
    batch_start: int,
    batch_stop: int,
    general_sequence_count: int,
    hard_math_sequence_count: int,
) -> tuple[int, int, int]:
    """Split Wiki, hard-math, and appended reasoning batches exactly."""

    if any(
        value < 0
        for value in (
            batch_start,
            batch_stop,
            general_sequence_count,
            hard_math_sequence_count,
        )
    ) or batch_stop < batch_start:
        raise ValueError("pareto domain boundaries must be non-negative and ordered")
    hard_stop = general_sequence_count + hard_math_sequence_count
    general_count = domain_general_prefix_count(
        batch_start, batch_stop, general_sequence_count
    )
    non_reasoning_count = domain_general_prefix_count(
        batch_start, batch_stop, hard_stop
    )
    return (
        general_count,
        non_reasoning_count - general_count,
        batch_stop - batch_start - non_reasoning_count,
    )


def layer_budget_prune_ratio(
    schedule: str,
    layer_index: int,
    layer_count: int,
    pruning_rate: float = PRUNE_RATIO,
) -> float:
    """Return deterministic layer ratios for a frozen global pruning endpoint."""

    if layer_count <= 0 or not 0 <= layer_index < layer_count:
        raise ValueError("layer index must be inside a non-empty model")
    rate = Fraction(str(pruning_rate))
    if rate not in {Fraction(1, 4), Fraction(1, 2)}:
        raise ValueError("pruning rate must be one of the frozen 0.25/0.50 endpoints")
    if schedule == "uniform":
        return float(rate)
    if schedule == "tail_protected":
        if layer_count % 3:
            raise ValueError("tail_protected requires a layer count divisible by 3")
        tail_start = 2 * layer_count // 3
        base_retention = (
            Fraction(7, 16) if layer_index < tail_start else Fraction(10, 16)
        )
        normalized_retention = 2 * (1 - rate) * base_retention
        if not 0 <= normalized_retention <= 1:
            raise ValueError("normalized tail retention lies outside [0, 1]")
        return float(1 - normalized_retention)
    raise ValueError(f"unsupported layer budget schedule: {schedule}")


def layer_capacity_aggregation(
    aggregation: str, layer_index: int, layer_count: int
) -> str:
    """Resolve a depth-conditioned capacity signal without changing its formulas."""

    if layer_count <= 0 or not 0 <= layer_index < layer_count:
        raise ValueError("layer index must be inside a non-empty model")
    depth_conditioned = {
        "anchored_tail_difference_capacity": (
            "anchored_geometric_capacity",
            "anchored_difference_capacity",
        ),
        "anchored_early_tail_domain_robust": (
            "anchored_geometric_capacity",
            "domain_robust",
        ),
        "anchored_early_tail_solution_domain_robust": (
            "anchored_geometric_capacity",
            "domain_robust",
        ),
        "anchored_early_tail_phase_domain_robust": (
            "anchored_geometric_capacity",
            "phase_domain_robust",
        ),
        "anchored_early_tail_phase_capacity_envelope": (
            "anchored_geometric_capacity",
            "phase_capacity_envelope",
        ),
        "anchored_tail_code_complement_capacity": (
            "anchored_geometric_capacity",
            "capability_complement_capacity",
        ),
        "reasoning_early_tail_dual_specialist_capacity": (
            "anchored_geometric_capacity",
            "dual_specialist_capacity",
        ),
        "tail_pareto_residual_refinement": (
            "pareto_prefix_frozen",
            "pareto_residual_refinement",
        ),
    }
    if aggregation not in depth_conditioned:
        return aggregation
    if layer_count % 3:
        raise ValueError(
            f"{aggregation} requires a layer count divisible by 3"
        )
    tail_start = 2 * layer_count // 3
    early_aggregation, tail_aggregation = depth_conditioned[aggregation]
    return early_aggregation if layer_index < tail_start else tail_aggregation


def trajectory_field_layer_enabled(layer_index: int, layer_count: int) -> bool:
    """Enable the fixed transition field only in the final model third."""

    if layer_count <= 0 or not 0 <= layer_index < layer_count:
        raise ValueError("layer index must be inside a non-empty model")
    if layer_count % 3:
        raise ValueError("trajectory field requires a layer count divisible by 3")
    return layer_index >= 2 * layer_count // 3


def layer_statistics_token_mask(
    aggregation: str,
    layer_index: int,
    layer_count: int,
    statistics_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Use full-token early statistics and solution targets only in the tail."""

    if layer_count <= 0 or not 0 <= layer_index < layer_count:
        raise ValueError("layer index must be inside a non-empty model")
    if aggregation not in {
        "anchored_early_tail_solution_domain_robust",
        "anchored_early_tail_phase_domain_robust",
        "anchored_early_tail_phase_capacity_envelope",
    }:
        return statistics_token_mask
    if layer_count % 3:
        raise ValueError(
            f"{aggregation} requires a layer count divisible by 3"
        )
    tail_start = 2 * layer_count // 3
    if layer_index < tail_start:
        return torch.ones_like(statistics_token_mask, dtype=torch.bool)
    return statistics_token_mask


def format_reasoning_trace(problem: str, solution: str) -> str:
    """Format a train-split problem and solution as an instruction trace."""

    problem = str(problem).strip()
    solution = str(solution).strip()
    if not problem or not solution:
        raise ValueError("reasoning problem and solution must be non-empty")
    return (
        "<|im_start|>user\n"
        f"{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{solution}<|im_end|>\n"
    )


def format_code_trace(problem: str, solution: str, tests: list[str]) -> str:
    """Format an MBPP train example as a code-generation chat trace."""

    problem = str(problem).strip()
    solution = str(solution).strip()
    tests = [str(test).strip() for test in tests if str(test).strip()]
    if not problem or not solution or not tests:
        raise ValueError("code problem, solution, and tests must be non-empty")
    test_block = "\n".join(tests)
    return (
        "<|im_start|>user\n"
        f"{problem}\n\nYour solution must pass these tests:\n{test_block}\n"
        "Return a complete Python solution.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{solution}<|im_end|>\n"
    )


def assistant_solution_token_mask(
    encoded: torch.Tensor,
    *,
    assistant_marker: torch.Tensor,
    end_marker: torch.Tensor,
) -> torch.Tensor:
    """Select predictor positions for assistant solution and end targets."""

    encoded = torch.as_tensor(encoded, dtype=torch.long).flatten()
    assistant_marker = torch.as_tensor(
        assistant_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    end_marker = torch.as_tensor(
        end_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    if assistant_marker.numel() == 0 or end_marker.numel() == 0:
        raise ValueError("assistant and end markers must be non-empty")

    def matching_starts(pattern: torch.Tensor) -> list[int]:
        candidates = torch.nonzero(encoded == pattern[0], as_tuple=False).flatten()
        return [
            int(index)
            for index in candidates.tolist()
            if index + pattern.numel() <= encoded.numel()
            and torch.equal(encoded[index : index + pattern.numel()], pattern)
        ]

    assistant_starts = matching_starts(assistant_marker)
    end_starts = matching_starts(end_marker)
    if not assistant_starts:
        raise ValueError("calibration trace has no assistant marker")
    target_mask = torch.zeros(
        encoded.shape, dtype=torch.bool, device=encoded.device
    )
    end_cursor = 0
    for marker_start in assistant_starts:
        content_start = marker_start + assistant_marker.numel()
        while end_cursor < len(end_starts) and end_starts[end_cursor] < content_start:
            end_cursor += 1
        if end_cursor == len(end_starts):
            raise ValueError("assistant calibration span has no closing marker")
        content_stop = end_starts[end_cursor]
        target_mask[content_start : content_stop + end_marker.numel()] = True
        end_cursor += 1
    predictor_mask = torch.zeros_like(target_mask)
    predictor_mask[:-1] = target_mask[1:]
    if not predictor_mask.any():
        raise ValueError("assistant calibration spans contain no solution tokens")
    return predictor_mask


def assistant_end_predictor_token_mask(
    encoded: torch.Tensor,
    *,
    end_marker: torch.Tensor,
    assistant_solution_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract assistant-end predictor positions from solution support."""

    encoded = torch.as_tensor(encoded, dtype=torch.long)
    assistant_solution_mask = torch.as_tensor(
        assistant_solution_mask, dtype=torch.bool, device=encoded.device
    )
    if encoded.shape != assistant_solution_mask.shape or encoded.ndim not in {1, 2}:
        raise ValueError("encoded tokens and solution mask must share 1D/2D shape")
    end_marker = torch.as_tensor(
        end_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    if end_marker.numel() == 0:
        raise ValueError("end marker must be non-empty")
    rows = encoded.unsqueeze(0) if encoded.ndim == 1 else encoded
    support_rows = (
        assistant_solution_mask.unsqueeze(0)
        if assistant_solution_mask.ndim == 1
        else assistant_solution_mask
    )
    predictor_rows = torch.zeros_like(support_rows)
    marker_length = int(end_marker.numel())
    for row_index, row in enumerate(rows):
        candidates = torch.nonzero(
            row == end_marker[0], as_tuple=False
        ).flatten()
        for marker_start in candidates.tolist():
            if marker_start + marker_length > row.numel():
                continue
            if not torch.equal(
                row[marker_start : marker_start + marker_length], end_marker
            ):
                continue
            predictor_start = max(marker_start - 1, 0)
            predictor_stop = marker_start + marker_length - 1
            predictor_rows[
                row_index, predictor_start:predictor_stop
            ] = True
    predictor_rows &= support_rows
    return predictor_rows[0] if encoded.ndim == 1 else predictor_rows


def assistant_chat_end_predictor_token_mask(
    encoded: torch.Tensor,
    *,
    assistant_marker: torch.Tensor,
    end_marker: torch.Tensor,
) -> torch.Tensor:
    """Select complete assistant-end prediction spans visible in each window."""

    encoded = torch.as_tensor(encoded, dtype=torch.long)
    if encoded.ndim not in {1, 2}:
        raise ValueError("encoded chat tokens must be one- or two-dimensional")
    assistant_marker = torch.as_tensor(
        assistant_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    end_marker = torch.as_tensor(
        end_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    if assistant_marker.numel() == 0 or end_marker.numel() == 0:
        raise ValueError("assistant and end markers must be non-empty")
    rows = encoded.unsqueeze(0) if encoded.ndim == 1 else encoded
    result = torch.zeros_like(rows, dtype=torch.bool)

    def matching_starts(row: torch.Tensor, pattern: torch.Tensor) -> list[int]:
        candidates = torch.nonzero(
            row == pattern[0], as_tuple=False
        ).flatten()
        return [
            int(index)
            for index in candidates.tolist()
            if index + pattern.numel() <= row.numel()
            and torch.equal(row[index : index + pattern.numel()], pattern)
        ]

    for row_index, row in enumerate(rows):
        assistant_starts = matching_starts(row, assistant_marker)
        end_starts = matching_starts(row, end_marker)
        end_cursor = 0
        for assistant_start in assistant_starts:
            content_start = assistant_start + assistant_marker.numel()
            while (
                end_cursor < len(end_starts)
                and end_starts[end_cursor] < content_start
            ):
                end_cursor += 1
            if end_cursor == len(end_starts):
                break
            marker_start = end_starts[end_cursor]
            predictor_start = max(marker_start - 1, 0)
            predictor_stop = marker_start + end_marker.numel() - 1
            result[row_index, predictor_start:predictor_stop] = True
            end_cursor += 1
    return result[0] if encoded.ndim == 1 else result


def assistant_chat_response_predictor_token_mask(
    encoded: torch.Tensor,
    *,
    assistant_marker: torch.Tensor,
    end_marker: torch.Tensor,
) -> torch.Tensor:
    """Select predictor positions for complete assistant response spans."""

    encoded = torch.as_tensor(encoded, dtype=torch.long)
    if encoded.ndim not in {1, 2}:
        raise ValueError("encoded chat tokens must be one- or two-dimensional")
    assistant_marker = torch.as_tensor(
        assistant_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    end_marker = torch.as_tensor(
        end_marker, dtype=torch.long, device=encoded.device
    ).flatten()
    if assistant_marker.numel() == 0 or end_marker.numel() == 0:
        raise ValueError("assistant and end markers must be non-empty")
    rows = encoded.unsqueeze(0) if encoded.ndim == 1 else encoded
    target = torch.zeros_like(rows, dtype=torch.bool)

    def matching_starts(row: torch.Tensor, pattern: torch.Tensor) -> list[int]:
        candidates = torch.nonzero(
            row == pattern[0], as_tuple=False
        ).flatten()
        return [
            int(index)
            for index in candidates.tolist()
            if index + pattern.numel() <= row.numel()
            and torch.equal(row[index : index + pattern.numel()], pattern)
        ]

    for row_index, row in enumerate(rows):
        assistant_starts = matching_starts(row, assistant_marker)
        end_starts = matching_starts(row, end_marker)
        end_cursor = 0
        for assistant_start in assistant_starts:
            content_start = assistant_start + assistant_marker.numel()
            while (
                end_cursor < len(end_starts)
                and end_starts[end_cursor] < content_start
            ):
                end_cursor += 1
            if end_cursor == len(end_starts):
                break
            marker_start = end_starts[end_cursor]
            target[
                row_index,
                content_start : marker_start + end_marker.numel(),
            ] = True
            end_cursor += 1
    predictor = torch.zeros_like(target)
    predictor[:, :-1] = target[:, 1:]
    return predictor[0] if encoded.ndim == 1 else predictor


def trajectory_phase_token_masks(
    assistant_solution_mask: torch.Tensor,
    assistant_end_mask: torch.Tensor,
    *,
    general_sequence_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition the math suffix into prompt, continuation, and termination."""

    solution = torch.as_tensor(assistant_solution_mask, dtype=torch.bool)
    termination = torch.as_tensor(
        assistant_end_mask, dtype=torch.bool, device=solution.device
    )
    if solution.ndim != 2 or solution.shape != termination.shape:
        raise ValueError("trajectory masks must share a two-dimensional shape")
    if not 0 <= general_sequence_count <= solution.shape[0]:
        raise ValueError("general_sequence_count is outside the batch")
    if bool((termination & ~solution).any().item()):
        raise ValueError("termination support must be inside solution support")
    continuation = solution & ~termination
    problem = ~solution
    problem[:general_sequence_count] = False
    continuation[:general_sequence_count] = False
    termination = termination.clone()
    termination[:general_sequence_count] = False
    if bool(
        (
            (problem & continuation)
            | (problem & termination)
            | (continuation & termination)
        ).any().item()
    ):
        raise RuntimeError("trajectory phase masks overlap")
    return problem, continuation, termination


def _sample_calibration_windows(
    text: str,
    tokenizer: object,
    *,
    sample_count: int,
    sequence_length: int,
    generator: random.Random,
    statistics_region: str = "all",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], int]:
    encoded = tokenizer(text, return_tensors="pt").input_ids
    if statistics_region == "all":
        statistics_mask = torch.ones(
            encoded.shape[1], dtype=torch.bool, device=encoded.device
        )
    elif statistics_region == "assistant_solution":
        assistant_marker = tokenizer(
            "<|im_start|>assistant\n", return_tensors="pt"
        ).input_ids.flatten()
        end_marker = tokenizer(
            "<|im_end|>", return_tensors="pt"
        ).input_ids.flatten()
        statistics_mask = assistant_solution_token_mask(
            encoded.flatten(),
            assistant_marker=assistant_marker,
            end_marker=end_marker,
        )
    else:
        raise ValueError(f"unsupported statistics region: {statistics_region}")
    upper_bound = encoded.shape[1] - sequence_length - 1
    if upper_bound < 0:
        raise ValueError("calibration corpus is shorter than one sequence")
    offsets = [generator.randint(0, upper_bound) for _ in range(sample_count)]
    samples = [
        encoded[:, offset : offset + sequence_length] for offset in offsets
    ]
    statistics_masks = [
        statistics_mask[offset : offset + sequence_length].unsqueeze(0)
        for offset in offsets
    ]
    return samples, statistics_masks, offsets, int(encoded.shape[1])


def calibration_tokens(
    model_source: str,
    revision: str | None,
    *,
    nsamples: int,
    sequence_length: int,
    seed: int,
    calibration_corpus: str,
    local_files_only: bool,
    competition_math_source: Path | None = None,
    competition_math_exclusion_manifest: Path | None = None,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    allocation = allocate_calibration_samples(nsamples, calibration_corpus)
    wiki = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        download_config=dataset_download_config(local_files_only),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=revision,
        use_fast=False,
        local_files_only=local_files_only,
    )
    assistant_end_marker_token_ids = tokenizer(
        "<|im_end|>", return_tensors="pt"
    ).input_ids.flatten().tolist()
    assistant_marker_token_ids = tokenizer(
        "<|im_start|>assistant\n", return_tensors="pt"
    ).input_ids.flatten().tolist()
    generator = random.Random(seed)
    windows, statistics_windows, wiki_offsets, wiki_tokens = (
        _sample_calibration_windows(
        "\n\n".join(wiki["text"]),
        tokenizer,
        sample_count=allocation["wikitext"],
        sequence_length=sequence_length,
        generator=generator,
        )
    )

    components: list[dict[str, object]] = [
        {
            "name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "dataset_split": "train",
            "dataset_fingerprint": wiki._fingerprint,
            "sample_count": allocation["wikitext"],
            "encoded_tokens": wiki_tokens,
            "offsets": wiki_offsets,
            "statistics_tokens": int(
                sum(mask.sum().item() for mask in statistics_windows)
            ),
        }
    ]

    def append_mbpp_windows() -> None:
        mbpp = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        mbpp_text = "\n".join(
            format_code_trace(row["text"], row["code"], row["test_list"])
            for row in mbpp
        )
        source_windows, source_statistics, offsets, encoded_tokens = (
            _sample_calibration_windows(
                mbpp_text,
                tokenizer,
                sample_count=allocation["mbpp"],
                sequence_length=sequence_length,
                generator=generator,
            )
        )
        windows.extend(source_windows)
        statistics_windows.extend(source_statistics)
        components.append(
            {
                "name": "mbpp",
                "dataset_config": "full",
                "dataset_split": "train",
                "dataset_fingerprint": mbpp._fingerprint,
                "sample_count": allocation["mbpp"],
                "encoded_tokens": encoded_tokens,
                "offsets": offsets,
                "statistics_tokens": int(
                    sum(mask.sum().item() for mask in source_statistics)
                ),
            }
        )

    if calibration_corpus in {
        "reasoning_mix",
        "reasoning_code_mix",
        "reasoning_code_append",
        "reasoning_hardmath_code_append",
    }:
        gsm = load_dataset(
            "gsm8k",
            "main",
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        math_qa = load_dataset(
            "math_qa",
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        gsm_text = "\n".join(
            format_reasoning_trace(row["question"], row["answer"])
            for row in gsm
        )
        math_qa_text = "\n".join(
            format_reasoning_trace(
                f'{row["Problem"]}\nOptions: {row["options"]}',
                row["Rationale"],
            )
            for row in math_qa
        )
        for name, dataset, text, count in (
            ("gsm8k", gsm, gsm_text, allocation["gsm8k"]),
            ("math_qa", math_qa, math_qa_text, allocation["math_qa"]),
        ):
            source_windows, source_statistics, offsets, encoded_tokens = (
                _sample_calibration_windows(
                    text,
                    tokenizer,
                    sample_count=count,
                    sequence_length=sequence_length,
                    generator=generator,
                )
            )
            windows.extend(source_windows)
            statistics_windows.extend(source_statistics)
            components.append(
                {
                    "name": name,
                    "dataset_split": "train",
                    "dataset_fingerprint": dataset._fingerprint,
                    "sample_count": count,
                    "encoded_tokens": encoded_tokens,
                    "offsets": offsets,
                    "statistics_tokens": int(
                        sum(mask.sum().item() for mask in source_statistics)
                    ),
                }
            )
    if calibration_corpus in {"reasoning_code_mix", "reasoning_code_append"}:
        append_mbpp_windows()
    if calibration_corpus in COMPETITION_MATH_CORPORA:
        if (
            competition_math_source is None
            or competition_math_exclusion_manifest is None
        ):
            raise ValueError(
                "competition MATH calibration requires its source and exclusion manifest"
            )
        excluded_hashes = load_competition_math_exclusion_hashes(
            competition_math_exclusion_manifest, competition_math_source
        )
        competition_math = load_dataset(
            "parquet",
            data_files={"train": str(competition_math_source)},
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        required_columns = {"problem", "solution", "level", "type"}
        if not required_columns.issubset(competition_math.column_names):
            raise ValueError(
                "competition MATH train parquet is missing required columns"
            )
        hard_candidates = [
            row
            for row in competition_math
            if str(row["level"]).strip() in {"Level 4", "Level 5"}
        ]
        hard_hashes = {
            normalized_problem_sha256(row["problem"]) for row in hard_candidates
        }
        missing_hashes = excluded_hashes - hard_hashes
        if missing_hashes:
            raise ValueError(
                "competition MATH exclusion manifest contains hashes outside "
                "the hard-level source"
            )
        hard_rows = [
            row
            for row in hard_candidates
            if normalized_problem_sha256(row["problem"]) not in excluded_hashes
        ]
        categories = sorted({str(row["type"]).strip() for row in hard_rows})
        per_level_count = allocation["competition_math"] // 2
        category_allocation = allocate_balanced_categories(
            per_level_count, categories
        )
        for level in ("Level 4", "Level 5"):
            for category, count in category_allocation.items():
                category_text = "\n".join(
                    format_reasoning_trace(row["problem"], row["solution"])
                    for row in hard_rows
                    if str(row["level"]).strip() == level
                    and str(row["type"]).strip() == category
                )
                source_windows, source_statistics, offsets, encoded_tokens = (
                    _sample_calibration_windows(
                        category_text,
                        tokenizer,
                        sample_count=count,
                        sequence_length=sequence_length,
                        generator=generator,
                        statistics_region=(
                            "assistant_solution"
                            if calibration_corpus
                            in {
                                "competition_math_solution_mix",
                                "competition_math_solution_reasoning_append",
                            }
                            else "all"
                        ),
                    )
                )
                windows.extend(source_windows)
                statistics_windows.extend(source_statistics)
                components.append(
                    {
                        "name": "competition_math",
                        "category": category,
                        "level": level,
                        "dataset_split": "train",
                        "dataset_fingerprint": competition_math._fingerprint,
                        "source": str(competition_math_source.resolve()),
                        "exclusion_manifest": str(
                            competition_math_exclusion_manifest.resolve()
                        ),
                        "excluded_problem_count": len(excluded_hashes),
                        "sample_count": count,
                        "encoded_tokens": encoded_tokens,
                        "offsets": offsets,
                        "statistics_region": (
                            "assistant_solution"
                            if calibration_corpus
                            in {
                                "competition_math_solution_mix",
                                "competition_math_solution_reasoning_append",
                            }
                            else "all"
                        ),
                        "statistics_tokens": int(
                            sum(
                                mask.sum().item()
                                for mask in source_statistics
                            )
                        ),
                    }
                )
    if calibration_corpus == "competition_math_solution_reasoning_append":
        gsm = load_dataset(
            "gsm8k",
            "main",
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        math_qa = load_dataset(
            "math_qa",
            split="train",
            download_config=dataset_download_config(local_files_only),
        )
        reasoning_specs = (
            (
                "gsm8k",
                gsm,
                "\n".join(
                    format_reasoning_trace(row["question"], row["answer"])
                    for row in gsm
                ),
            ),
            (
                "math_qa",
                math_qa,
                "\n".join(
                    format_reasoning_trace(
                        f'{row["Problem"]}\nOptions: {row["options"]}',
                        row["Rationale"],
                    )
                    for row in math_qa
                ),
            ),
        )
        for name, dataset, text in reasoning_specs:
            source_windows, source_statistics, offsets, encoded_tokens = (
                _sample_calibration_windows(
                    text,
                    tokenizer,
                    sample_count=allocation[name],
                    sequence_length=sequence_length,
                    generator=generator,
                )
            )
            windows.extend(source_windows)
            statistics_windows.extend(source_statistics)
            components.append(
                {
                    "name": name,
                    "dataset_split": "train",
                    "dataset_fingerprint": dataset._fingerprint,
                    "sample_count": allocation[name],
                    "encoded_tokens": encoded_tokens,
                    "offsets": offsets,
                    "statistics_tokens": int(
                        sum(mask.sum().item() for mask in source_statistics)
                    ),
                }
            )
    if calibration_corpus in {
        "competition_math_code_append",
        "reasoning_hardmath_code_append",
    }:
        append_mbpp_windows()

    samples = torch.cat(windows, dim=0)
    statistics_token_mask = torch.cat(statistics_windows, dim=0)
    if statistics_token_mask.shape != samples.shape:
        raise RuntimeError("calibration statistics mask shape mismatch")
    metadata = {
        "dataset": calibration_corpus,
        "sample_allocation": allocation,
        "components": components,
        "seed": seed,
        "nsamples": nsamples,
        "sequence_length": sequence_length,
        "statistics_token_count": int(statistics_token_mask.sum().item()),
        "statistics_token_fraction": float(
            statistics_token_mask.float().mean().item()
        ),
        "assistant_end_marker_token_ids": (
            assistant_end_marker_token_ids
            if calibration_corpus
            in {
                "competition_math_solution_mix",
                "competition_math_solution_reasoning_append",
                "reasoning_hardmath_code_append",
            }
            else None
        ),
        "assistant_marker_token_ids": (
            assistant_marker_token_ids
            if calibration_corpus
            in {
                "competition_math_solution_mix",
                "competition_math_solution_reasoning_append",
                "reasoning_hardmath_code_append",
            }
            else None
        ),
    }
    return samples, metadata, statistics_token_mask


def _flat_token_ids(encoded: object) -> list[int]:
    values = getattr(encoded, "input_ids", encoded)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if isinstance(values, list) and len(values) == 1 and isinstance(
        values[0], list
    ):
        values = values[0]
    if not isinstance(values, list) or any(
        not isinstance(value, int) for value in values
    ):
        raise ValueError("tokenizer did not return a flat integer token list")
    return values


def append_loop_contrast_calibration(
    base_samples: torch.Tensor,
    base_statistics_mask: torch.Tensor,
    tokenizer: object,
    *,
    manifest_path: Path,
    sequence_length: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Append paired real-loop and gold branches with causal phase masks."""

    if base_samples.ndim != 2 or base_statistics_mask.shape != base_samples.shape:
        raise ValueError("base samples and statistics mask must share [N, L]")
    if base_samples.shape[1] != sequence_length:
        raise ValueError("base sample length does not match sequence_length")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    samples = payload.get("samples")
    summary = payload.get("summary")
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_type")
        != "train_only_self_induced_loop_mining"
        or not isinstance(protocol, dict)
        or not isinstance(samples, list)
        or not isinstance(summary, dict)
    ):
        raise ValueError("loop manifest has an incompatible schema")
    expected_protocol = {
        "temperature": 0,
        "causal_length": 2048,
        "ngram_size": 8,
        "repetition_threshold": 0.25,
        "seed": 1234,
        "test_outputs_read": False,
        "synthetic_loop_fallback": False,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("loop manifest violates the fixed mining protocol")
    if summary.get("samples") != len(samples):
        raise ValueError("loop manifest sample count does not match its rows")
    positive_rows = [row for row in samples if bool(row.get("high_repetition"))]
    if summary.get("high_repetition") != len(positive_rows) or not positive_rows:
        raise ValueError("loop manifest has no consistent positive loop rows")

    end_ids = _flat_token_ids(
        tokenizer("<|im_end|>", add_special_tokens=False)
    )
    if not end_ids:
        raise ValueError("tokenizer returned no assistant-end marker")
    if protocol.get("reserved_assistant_end_tokens") != len(end_ids):
        raise ValueError(
            "loop manifest did not reserve the tokenizer assistant-end marker"
        )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, int):
        raise ValueError("tokenizer must define an integer pad or EOS token")

    loop_branches: list[torch.Tensor] = []
    gold_branches: list[torch.Tensor] = []
    loop_masks: list[torch.Tensor] = []
    gold_masks: list[torch.Tensor] = []
    pair_records: list[dict[str, object]] = []
    discarded: list[dict[str, object]] = []
    for row in positive_rows:
        required_strings = (
            "domain",
            "source",
            "row_id",
            "prompt",
            "gold_response",
            "model_output",
        )
        if any(not isinstance(row.get(key), str) for key in required_strings):
            raise ValueError("loop manifest row is missing required text fields")
        prompt = str(row["prompt"])
        gold = str(row["gold_response"])
        output = str(row["model_output"])
        for field, text in (
            ("prompt", prompt),
            ("gold_response", gold),
            ("model_output", output),
        ):
            expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if row.get(f"{field}_sha256") != expected_hash:
                raise ValueError(f"loop manifest {field} SHA256 mismatch")
        repetition = repeated_ngram_fraction(output, n=8)
        occurrence = first_repeated_ngram_occurrence(output, n=8)
        if (
            repetition < 0.25
            or abs(repetition - float(row.get("repeated_ngram_fraction", -1)))
            > 1e-12
            or occurrence != row.get("first_repeated_ngram")
            or occurrence is None
        ):
            raise ValueError("loop manifest repetition evidence does not reproduce")
        prompt_ids = _flat_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        output_ids = _flat_token_ids(
            tokenizer(output, add_special_tokens=False)
        )
        gold_ids = _flat_token_ids(tokenizer(gold, add_special_tokens=False))
        onset_character = int(occurrence["repeated_character_start"])
        prefix_ids = _flat_token_ids(
            tokenizer(
                output[:onset_character].rstrip(),
                add_special_tokens=False,
            )
        )
        identity = f'{row["source"]}:{row["row_id"]}'
        if output_ids[: len(prefix_ids)] != prefix_ids:
            discarded.append(
                {"identity": identity, "reason": "unstable_token_prefix"}
            )
            continue
        loop_ids = prompt_ids + output_ids + end_ids
        gold_branch_ids = prompt_ids + gold_ids + end_ids
        if len(loop_ids) > sequence_length or len(gold_branch_ids) > sequence_length:
            discarded.append(
                {
                    "identity": identity,
                    "reason": "branch_exceeds_causal_length",
                    "loop_tokens": len(loop_ids),
                    "gold_tokens": len(gold_branch_ids),
                }
            )
            continue
        loop_target_start = len(prompt_ids) + len(prefix_ids)
        loop_target_stop = len(loop_ids)
        gold_target_start = len(prompt_ids)
        gold_target_stop = len(gold_branch_ids)
        if loop_target_start <= 0 or gold_target_start <= 0:
            raise RuntimeError("assistant targets have no causal predictor")

        loop_mask = torch.zeros(sequence_length, dtype=torch.bool)
        gold_mask = torch.zeros(sequence_length, dtype=torch.bool)
        loop_mask[loop_target_start - 1 : loop_target_stop - 1] = True
        gold_mask[gold_target_start - 1 : gold_target_stop - 1] = True
        if not bool(loop_mask.any()) or not bool(gold_mask.any()):
            raise RuntimeError("loop contrast pair has an empty causal phase")
        loop_branches.append(
            torch.tensor(
                loop_ids + [pad_id] * (sequence_length - len(loop_ids)),
                dtype=base_samples.dtype,
            ).unsqueeze(0)
        )
        gold_branches.append(
            torch.tensor(
                gold_branch_ids
                + [pad_id] * (sequence_length - len(gold_branch_ids)),
                dtype=base_samples.dtype,
            ).unsqueeze(0)
        )
        loop_masks.append(loop_mask.unsqueeze(0))
        gold_masks.append(gold_mask.unsqueeze(0))
        pair_records.append(
            {
                "identity": identity,
                "domain": row["domain"],
                "prompt_sha256": row["prompt_sha256"],
                "model_output_sha256": row["model_output_sha256"],
                "gold_response_sha256": row["gold_response_sha256"],
                "loop_branch_tokens": len(loop_ids),
                "gold_branch_tokens": len(gold_branch_ids),
                "loop_statistics_tokens": int(loop_mask.sum().item()),
                "gold_statistics_tokens": int(gold_mask.sum().item()),
            }
        )
    if not pair_records:
        raise RuntimeError("no loop contrast pair preserves the full causal prefix")

    loop_tensor = torch.cat(loop_branches, dim=0)
    gold_tensor = torch.cat(gold_branches, dim=0)
    pair_count = len(pair_records)
    extended_samples = torch.cat((base_samples, loop_tensor, gold_tensor), dim=0)
    extended_statistics = torch.cat(
        (
            base_statistics_mask,
            torch.zeros(
                (2 * pair_count, sequence_length), dtype=torch.bool
            ),
        ),
        dim=0,
    )
    loop_selection = torch.zeros_like(extended_statistics)
    gold_selection = torch.zeros_like(extended_statistics)
    base_count = base_samples.shape[0]
    loop_selection[base_count : base_count + pair_count] = torch.cat(
        loop_masks, dim=0
    )
    gold_selection[base_count + pair_count :] = torch.cat(
        gold_masks, dim=0
    )
    if bool((loop_selection & gold_selection).any().item()):
        raise RuntimeError("loop and gold phase masks overlap")
    metadata = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_sequence_count": base_count,
        "valid_pair_count": pair_count,
        "discarded_pair_count": len(discarded),
        "discarded_pairs": discarded,
        "pairs": pair_records,
    }
    return (
        extended_samples,
        extended_statistics,
        loop_selection,
        gold_selection,
        metadata,
    )


def append_loop_onset_calibration(
    base_samples: torch.Tensor,
    base_statistics_mask: torch.Tensor,
    tokenizer: object,
    *,
    manifest_path: Path,
    sequence_length: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    """Append one causally aligned predictor prefix per real train loop."""

    if base_samples.ndim != 2 or base_statistics_mask.shape != base_samples.shape:
        raise ValueError("base samples and statistics mask must share [N, L]")
    if base_samples.shape[1] != sequence_length:
        raise ValueError("base sample length does not match sequence_length")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    samples = payload.get("samples")
    summary = payload.get("summary")
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_type")
        != "train_only_self_induced_loop_mining"
        or not isinstance(protocol, dict)
        or not isinstance(samples, list)
        or not isinstance(summary, dict)
    ):
        raise ValueError("loop manifest has an incompatible schema")
    expected_protocol = {
        "temperature": 0,
        "causal_length": 2048,
        "ngram_size": 8,
        "repetition_threshold": 0.25,
        "seed": 1234,
        "test_outputs_read": False,
        "synthetic_loop_fallback": False,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("loop manifest violates the fixed mining protocol")
    if summary.get("samples") != len(samples):
        raise ValueError("loop manifest sample count does not match its rows")
    positive_rows = [row for row in samples if bool(row.get("high_repetition"))]
    if summary.get("high_repetition") != len(positive_rows) or not positive_rows:
        raise ValueError("loop manifest has no consistent positive loop rows")

    end_ids = _flat_token_ids(
        tokenizer("<|im_end|>", add_special_tokens=False)
    )
    if not end_ids:
        raise ValueError("tokenizer returned no assistant-end marker")
    if protocol.get("reserved_assistant_end_tokens") != len(end_ids):
        raise ValueError(
            "loop manifest did not reserve the tokenizer assistant-end marker"
        )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, int):
        raise ValueError("tokenizer must define an integer pad or EOS token")

    context_branches: list[torch.Tensor] = []
    context_masks: list[torch.Tensor] = []
    context_records: list[dict[str, object]] = []
    discarded: list[dict[str, object]] = []
    for row in positive_rows:
        required_strings = (
            "domain",
            "source",
            "row_id",
            "prompt",
            "gold_response",
            "model_output",
        )
        if any(not isinstance(row.get(key), str) for key in required_strings):
            raise ValueError("loop manifest row is missing required text fields")
        prompt = str(row["prompt"])
        gold = str(row["gold_response"])
        output = str(row["model_output"])
        for field, text in (
            ("prompt", prompt),
            ("gold_response", gold),
            ("model_output", output),
        ):
            expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if row.get(f"{field}_sha256") != expected_hash:
                raise ValueError(f"loop manifest {field} SHA256 mismatch")
        repetition = repeated_ngram_fraction(output, n=8)
        occurrence = first_repeated_ngram_occurrence(output, n=8)
        if (
            repetition < 0.25
            or abs(repetition - float(row.get("repeated_ngram_fraction", -1)))
            > 1e-12
            or occurrence != row.get("first_repeated_ngram")
            or occurrence is None
        ):
            raise ValueError("loop manifest repetition evidence does not reproduce")
        prompt_ids = _flat_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        output_ids = _flat_token_ids(
            tokenizer(output, add_special_tokens=False)
        )
        onset_character = int(occurrence["repeated_character_start"])
        prefix_ids = _flat_token_ids(
            tokenizer(
                output[:onset_character].rstrip(),
                add_special_tokens=False,
            )
        )
        identity = f'{row["source"]}:{row["row_id"]}'
        if output_ids[: len(prefix_ids)] != prefix_ids:
            discarded.append(
                {"identity": identity, "reason": "unstable_token_prefix"}
            )
            continue
        if not prefix_ids or len(prefix_ids) >= len(output_ids):
            discarded.append(
                {"identity": identity, "reason": "missing_predictor_target"}
            )
            continue
        branch_ids = prompt_ids + prefix_ids
        if len(branch_ids) > sequence_length:
            discarded.append(
                {
                    "identity": identity,
                    "reason": "context_exceeds_causal_length",
                    "context_tokens": len(branch_ids),
                }
            )
            continue
        predictor_index = len(branch_ids) - 1
        context_mask = torch.zeros(sequence_length, dtype=torch.bool)
        context_mask[predictor_index] = True
        context_branches.append(
            torch.tensor(
                branch_ids + [pad_id] * (sequence_length - len(branch_ids)),
                dtype=base_samples.dtype,
            ).unsqueeze(0)
        )
        context_masks.append(context_mask.unsqueeze(0))
        input_sha256 = hashlib.sha256(
            json.dumps(branch_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        context_records.append(
            {
                "identity": identity,
                "domain": row["domain"],
                "prompt_sha256": row["prompt_sha256"],
                "model_output_sha256": row["model_output_sha256"],
                "input_ids_sha256": input_sha256,
                "context_tokens": len(branch_ids),
                "predictor_index": predictor_index,
                "target_token_id": int(output_ids[len(prefix_ids)]),
            }
        )
    if not context_records:
        raise RuntimeError("no loop onset preserves a stable causal predictor")

    context_tensor = torch.cat(context_branches, dim=0)
    extended_samples = torch.cat((base_samples, context_tensor), dim=0)
    extended_statistics = torch.cat(
        (
            base_statistics_mask,
            torch.zeros_like(context_tensor, dtype=torch.bool),
        ),
        dim=0,
    )
    onset_selection = torch.zeros_like(extended_statistics)
    base_count = base_samples.shape[0]
    onset_selection[base_count:] = torch.cat(context_masks, dim=0)
    domain_counts: dict[str, int] = {}
    for record in context_records:
        domain = str(record["domain"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    metadata = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_sequence_count": base_count,
        "valid_context_count": len(context_records),
        "discarded_context_count": len(discarded),
        "discarded_contexts": discarded,
        "domain_counts": domain_counts,
        "contexts": context_records,
    }
    return extended_samples, extended_statistics, onset_selection, metadata


@torch.no_grad()
def capture_dense_loop_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    predictor_mask: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Capture dense router-weighted channel states at aligned predictors."""

    if input_ids.ndim != 2 or predictor_mask.shape != input_ids.shape:
        raise ValueError("input_ids and predictor_mask must share [N, L]")
    if input_ids.shape[0] == 0:
        raise ValueError("dense loop-state capture requires at least one sample")
    selected = predictor_mask.to(dtype=torch.bool)
    if not torch.equal(
        selected.sum(dim=1), torch.ones(input_ids.shape[0], dtype=torch.long)
    ):
        raise ValueError("each dense loop-state sample needs one predictor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    layers = resolve_decoder_layers(model)
    if not layers:
        raise ValueError("model must expose at least one decoder layer")
    first_moe = layers[0].mlp
    expert_count = len(first_moe.experts)
    channel_count = first_moe.experts[0].up_proj.weight.shape[0]
    states: list[torch.Tensor] = []
    routed_evidence = [0 for _ in layers]
    dense_inputs, layer_kwargs = capture_first_layer_inputs(
        model,
        input_ids,
        device=device,
        chunk_size=chunk_size,
    )
    dense_outputs = torch.empty_like(dense_inputs)
    selected = selected.to(device)
    for layer_index, layer in enumerate(layers):
        layer.to(device)
        layer_states = torch.zeros(
            (input_ids.shape[0], expert_count, channel_count),
            dtype=torch.bfloat16,
        )
        for start in range(0, input_ids.shape[0], chunk_size):
            stop = min(start + chunk_size, input_ids.shape[0])
            means, routed, selected_counts = router_weighted_phase_means(
                layer.mlp,
                dense_inputs[start:stop],
                token_selection_mask=selected[start:stop],
            )
            if (selected_counts != 1).any():
                raise RuntimeError("dense loop-state predictor selection drifted")
            layer_states[start:stop].copy_(
                means.detach().to(device="cpu", dtype=torch.bfloat16)
            )
            routed_evidence[layer_index] += int(routed.sum().item())
            dense_outputs[start:stop] = decoder_layer_hidden(
                layer,
                dense_inputs[start:stop],
                layer_kwargs,
                layer_index=layer_index,
                batch_start=start,
                replay_pass="dense",
                captured_batch_size=min(chunk_size, input_ids.shape[0]),
            )
        states.append(layer_states)
        layer.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dense_inputs, dense_outputs = dense_outputs, dense_inputs
    if any(value <= 0 for value in routed_evidence):
        raise RuntimeError("dense loop-state capture has an empty layer")
    return states, {
        "samples": input_ids.shape[0],
        "selected_predictors": int(selected.sum().item()),
        "layers": len(layers),
        "expert_count": expert_count,
        "channel_count": channel_count,
        "chunk_size": chunk_size,
        "storage_dtype": "bfloat16",
        "routed_evidence_per_layer": routed_evidence,
    }


@torch.no_grad()
def capture_first_layer_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, LayerReplayContext]:
    layers = resolve_decoder_layers(model)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError("decoder layers must be mutable for input capture")
    embedding = resolve_input_embeddings(model)
    embedding_weight = getattr(embedding, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or embedding_weight.ndim != 2:
        raise TypeError("input embedding module must expose a rank-two weight")
    embedding.to(device)
    dtype = embedding_weight.dtype
    inputs = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embedding_weight.shape[1]),
        dtype=dtype,
        device=device,
    )
    state: dict[str, object] = {
        "offset": 0,
        "layer_calls": [None for _ in layers],
    }

    class Catcher(torch.nn.Module):
        def __init__(self, layer_index: int) -> None:
            super().__init__()
            self.layer_index = layer_index

        def forward(
            self,
            hidden_states: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> torch.Tensor:
            if self.layer_index == 0:
                offset = int(state["offset"])
                batch_size = hidden_states.shape[0]
                inputs[offset : offset + batch_size] = hidden_states
                state["offset"] = offset + batch_size
            layer_calls = state["layer_calls"]
            if layer_calls[self.layer_index] is None:
                allowed = {
                    "attention_mask",
                    "position_ids",
                    "position_embeddings",
                    "cache_position",
                    "shared_kv_states",
                    "past_key_values",
                    "use_cache",
                }
                layer_calls[self.layer_index] = _CapturedLayerCall(
                    args=tuple(args),
                    kwargs={
                        key: value
                        for key, value in kwargs.items()
                        if key in allowed
                    },
                )
            if self.layer_index + 1 == len(layers):
                raise _CaptureComplete
            return hidden_states

    original_layers = list(layers)
    for layer_index in range(len(layers)):
        layers[layer_index] = Catcher(layer_index)
    try:
        for start in range(0, input_ids.shape[0], chunk_size):
            batch = input_ids[start : start + chunk_size].to(device)
            try:
                model(input_ids=batch, use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        for layer_index, layer in enumerate(original_layers):
            layers[layer_index] = layer
        embedding.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if state["offset"] != input_ids.shape[0]:
        raise RuntimeError("failed to capture every calibration sequence")
    layer_calls = state["layer_calls"]
    if any(call is None for call in layer_calls):
        raise RuntimeError("failed to capture every decoder layer call")
    return inputs, LayerReplayContext(calls=tuple(layer_calls))


def _uniform_method_mask(
    accumulator: Qwen3LayerStatsAccumulator,
    *,
    method: str,
    prune_ratio: float,
) -> tuple[torch.Tensor, list[dict[str, object]], None]:
    exact_keep_count = Fraction(accumulator.channel_count) * (
        1 - Fraction(str(prune_ratio))
    )
    if exact_keep_count.denominator != 1:
        raise ValueError("expert width cannot represent pruning endpoint exactly")
    keep_count = exact_keep_count.numerator
    mask = torch.zeros(
        (accumulator.expert_count, accumulator.channel_count),
        dtype=torch.bool,
        device=accumulator.device,
    )
    records: list[dict[str, object]] = []

    for expert_index in range(accumulator.expert_count):
        gram = accumulator.interaction_gram(expert_index)
        diagonal_keep = diagonal_keep_indices(gram, keep_count)
        ramp_keep = forward_residual_keep_indices(gram, keep_count)
        selected = diagonal_keep if method == "diagonal" else ramp_keep
        mask[expert_index, selected] = True

        diagonal_error = gram_reconstruction_error(gram, diagonal_keep)
        ramp_error = gram_reconstruction_error(gram, ramp_keep)
        diagonal = gram.diag().clamp_min(0)
        denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :]).clamp_min(
            1e-12
        )
        correlation = gram / denominator
        off_diagonal = ~torch.eye(
            gram.shape[0], dtype=torch.bool, device=gram.device
        )
        off_values = correlation[off_diagonal]
        records.append(
            {
                "expert": expert_index,
                "routed_tokens": int(
                    accumulator.routed_token_counts[expert_index].item()
                ),
                "reservoir_tokens": accumulator.reservoir_size_for(expert_index),
                "selection_status": (
                    "measured"
                    if accumulator.reservoir_size_for(expert_index) > 0
                    else "unrouted_zero_gram_tie"
                ),
                "original_output_energy": float(gram.sum().item()),
                "diagonal_reconstruction_error": float(diagonal_error.item()),
                "ramp_reconstruction_error": float(ramp_error.item()),
                "ramp_relative_improvement": float(
                    ((diagonal_error - ramp_error) / diagonal_error.clamp_min(1e-12)).item()
                ),
                "mean_abs_corr": float(off_values.abs().mean().item()),
                "negative_corr_ratio": float((off_values < 0).float().mean().item()),
            }
        )
    return mask, records, None


def select_ramp_camera_mask_with_expert_widths(
    accumulator: Qwen3LayerStatsAccumulator,
    expert_widths: torch.Tensor,
    *,
    camera_alpha: float,
    interaction_strength: float,
    max_swaps: int,
    coefficient_ridge_ratio: float | None = None,
    coefficient_clip_min: float = 0.5,
    coefficient_clip_max: float = 1.5,
    initial_keep_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]], torch.Tensor | None]:
    """Run RAMP refinement after a model-wide allocator fixes expert widths."""

    importance = accumulator.camera_importance(camera_alpha)
    widths = torch.as_tensor(expert_widths, device=importance.device)
    if widths.shape != (accumulator.expert_count,):
        raise ValueError("expert_widths must contain one value per expert")
    if widths.dtype == torch.bool or widths.is_floating_point():
        raise ValueError("expert_widths must contain integers")
    widths = widths.to(torch.long)
    if (widths < 0).any() or (widths > accumulator.channel_count).any():
        raise ValueError("expert_widths must lie inside the channel range")
    if initial_keep_mask is None:
        camera_mask = select_keep_mask_with_expert_widths(importance, widths)
        initialization = "current_trajectory_camera_topk"
    else:
        camera_mask = torch.as_tensor(
            initial_keep_mask, device=importance.device
        )
        if camera_mask.dtype != torch.bool or camera_mask.shape != importance.shape:
            raise ValueError(
                "initial_keep_mask must be a boolean expert-by-channel tensor"
            )
        camera_mask = camera_mask.clone()
        if not torch.equal(camera_mask.sum(dim=1), widths):
            raise ValueError("initial_keep_mask must match the fixed expert widths")
        initialization = "frozen_stage1_mask"
    refined_mask = camera_mask.clone()
    records: list[dict[str, object]] = []
    coefficients = (
        torch.ones_like(importance)
        if coefficient_ridge_ratio is not None
        else None
    )

    for expert_index in range(accumulator.expert_count):
        initial = torch.nonzero(
            camera_mask[expert_index], as_tuple=False
        ).flatten()
        keep_count = initial.numel()
        raw_gram = accumulator.interaction_gram(expert_index)
        robust_gram = reweight_gram_with_importance(
            raw_gram,
            importance[expert_index],
            interaction_strength=interaction_strength,
        )

        if keep_count == 0:
            refined = initial
            initial_error = robust_gram.sum()
            refined_error = initial_error
            status = "camera_all_pruned"
        elif keep_count == accumulator.channel_count:
            refined = initial
            initial_error = torch.zeros((), device=robust_gram.device)
            refined_error = initial_error
            status = "camera_all_retained"
        else:
            initial_error = gram_reconstruction_error(robust_gram, initial)
            refined = swap_refine_keep_indices(
                robust_gram, initial, max_swaps=max_swaps
            )
            refined_error = gram_reconstruction_error(robust_gram, refined)
            status = (
                "refined"
                if not torch.equal(torch.sort(initial).values, refined)
                else "no_beneficial_swap"
            )
            refined_mask[expert_index] = False
            refined_mask[expert_index, refined] = True

        swap_count = int(
            (camera_mask[expert_index] != refined_mask[expert_index])
            .sum()
            .item()
            // 2
        )
        recovered = None
        if coefficients is not None and keep_count not in {
            0,
            accumulator.channel_count,
        }:
            recovery_gram = accumulator.reconstruction_gram(expert_index)
            recovered = recover_channel_coefficients(
                recovery_gram,
                refined,
                ridge_ratio=coefficient_ridge_ratio,
                clip_min=coefficient_clip_min,
                clip_max=coefficient_clip_max,
            )
            coefficients[expert_index, refined] = recovered
        records.append(
            {
                "expert": expert_index,
                "routed_tokens": int(
                    accumulator.routed_token_counts[expert_index].item()
                ),
                "reservoir_tokens": accumulator.reservoir_size_for(expert_index),
                "retained_channels": keep_count,
                "initialization": initialization,
                "selection_status": status,
                "swap_count": swap_count,
                "camera_objective_error": float(initial_error.item()),
                "refined_objective_error": float(refined_error.item()),
                "relative_objective_improvement": float(
                    (
                        (initial_error - refined_error)
                        / initial_error.abs().clamp_min(1e-12)
                    ).item()
                ),
                "raw_interaction_energy": float(raw_gram.sum().item()),
                "coefficient_min": (
                    None if recovered is None else float(recovered.min().item())
                ),
                "coefficient_max": (
                    None if recovered is None else float(recovered.max().item())
                ),
                "coefficient_mean": (
                    None if recovered is None else float(recovered.mean().item())
                ),
            }
        )

    if not torch.equal(refined_mask.sum(dim=1), widths):
        raise RuntimeError("RAMP changed the preallocated expert widths")
    return refined_mask, records, coefficients


def _ramp_camera_global_mask(
    accumulator: Qwen3LayerStatsAccumulator,
    *,
    camera_alpha: float,
    interaction_strength: float,
    max_swaps: int,
    min_channels_per_expert: int,
    coefficient_ridge_ratio: float | None,
    coefficient_clip_min: float,
    coefficient_clip_max: float,
    prune_ratio: float,
    max_width_delta_divisor: int = 8,
) -> tuple[torch.Tensor, list[dict[str, object]], torch.Tensor | None]:
    if (
        isinstance(max_width_delta_divisor, bool)
        or not isinstance(max_width_delta_divisor, int)
        or max_width_delta_divisor <= 0
    ):
        raise ValueError("max width delta divisor must be a positive integer")
    importance = accumulator.camera_importance(camera_alpha)
    allocation_importance_fn = getattr(accumulator, "allocation_importance", None)
    allocation_importance = (
        allocation_importance_fn(camera_alpha)
        if callable(allocation_importance_fn)
        else importance
    )
    allocation_reference_fn = getattr(
        accumulator, "allocation_reference_importance", None
    )
    allocation_reference = (
        allocation_reference_fn(camera_alpha)
        if callable(allocation_reference_fn)
        else None
    )
    if allocation_reference is None:
        camera_mask = select_camera_keep_mask(
            allocation_importance,
            prune_ratio=prune_ratio,
            allocation="global",
            min_keep_per_expert=min_channels_per_expert,
        )
    else:
        uniform_keep_count = round(
            accumulator.channel_count * (1 - prune_ratio)
        )
        max_width_delta = max(1, uniform_keep_count // max_width_delta_divisor)
        reference_width_target_fn = getattr(
            accumulator,
            "allocation_importance_for_reference_widths",
            None,
        )
        if callable(reference_width_target_fn):
            reference_mask = select_camera_keep_mask(
                allocation_reference,
                prune_ratio=prune_ratio,
                allocation="global",
                min_keep_per_expert=min_channels_per_expert,
            )
            allocation_importance = reference_width_target_fn(
                camera_alpha, reference_mask.sum(dim=1)
            )
        camera_mask = select_bounded_camera_keep_mask(
            allocation_importance,
            reference_importance=allocation_reference,
            prune_ratio=prune_ratio,
            max_width_delta=max_width_delta,
            min_keep_per_expert=min_channels_per_expert,
        )
    refined_mask, records, coefficients = select_ramp_camera_mask_with_expert_widths(
        accumulator,
        camera_mask.sum(dim=1),
        camera_alpha=camera_alpha,
        interaction_strength=interaction_strength,
        max_swaps=max_swaps,
        coefficient_ridge_ratio=coefficient_ridge_ratio,
        coefficient_clip_min=coefficient_clip_min,
        coefficient_clip_max=coefficient_clip_max,
    )
    multiplier_fn = getattr(
        accumulator, "capacity_multipliers_for_reference_widths", None
    )
    if allocation_reference is not None and callable(multiplier_fn):
        reference_widths = reference_mask.sum(dim=1)
        capacity_multipliers = multiplier_fn(camera_alpha, reference_widths)
        for expert_index, record in enumerate(records):
            reference = torch.nonzero(
                reference_mask[expert_index], as_tuple=False
            ).flatten()
            raw_gram = accumulator.interaction_gram(expert_index)
            robust_gram = reweight_gram_with_importance(
                raw_gram,
                importance[expert_index],
                interaction_strength=interaction_strength,
            )
            reference_error = gram_reconstruction_error(robust_gram, reference)
            current_error = float(record["camera_objective_error"])
            record.update(
                {
                    "reference_retained_channels": int(
                        reference_widths[expert_index].item()
                    ),
                    "capacity_multiplier": float(
                        capacity_multipliers[expert_index].item()
                    ),
                    "capacity_width_delta": int(
                        camera_mask[expert_index].sum().item()
                        - reference_widths[expert_index].item()
                    ),
                    "reference_objective_error": float(reference_error.item()),
                    "capacity_objective_improvement": float(
                        reference_error.item() - current_error
                    ),
                }
            )
    return refined_mask, records, coefficients


def select_layer_mask(
    accumulator: Qwen3LayerStatsAccumulator,
    *,
    method: str,
    allocation: str,
    camera_alpha: float,
    interaction_strength: float,
    max_swaps: int,
    min_channels_per_expert: int = 0,
    coefficient_ridge_ratio: float | None = None,
    coefficient_clip_min: float = 0.5,
    coefficient_clip_max: float = 1.5,
    prune_ratio: float = PRUNE_RATIO,
    max_width_delta_divisor: int = 8,
) -> tuple[
    torch.Tensor, list[dict[str, object]], torch.Tensor | None
]:
    if method == "camera":
        importance = accumulator.camera_importance(camera_alpha)
        return (
            select_camera_keep_mask(
                importance,
                prune_ratio=prune_ratio,
                allocation=allocation,
                min_keep_per_expert=min_channels_per_expert,
            ),
            [],
            None,
        )
    if method == "ramp_camera":
        return _ramp_camera_global_mask(
            accumulator,
            camera_alpha=camera_alpha,
            interaction_strength=interaction_strength,
            max_swaps=max_swaps,
            min_channels_per_expert=min_channels_per_expert,
            coefficient_ridge_ratio=coefficient_ridge_ratio,
            coefficient_clip_min=coefficient_clip_min,
            coefficient_clip_max=coefficient_clip_max,
            prune_ratio=prune_ratio,
            max_width_delta_divisor=max_width_delta_divisor,
        )
    return _uniform_method_mask(
        accumulator,
        method=method,
        prune_ratio=prune_ratio,
    )


def make_stats_hook(accumulator: Qwen3LayerStatsAccumulator):
    def collect_stats(
        _module: torch.nn.Module, hook_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        accumulator.update(hook_inputs[0])

    return collect_stats


def next_output_metric_scale(
    model: torch.nn.Module, layer_index: int, output_metric: str
) -> torch.Tensor | None:
    """Return the next RMSNorm gamma for a diagonal output metric."""

    if output_metric == "euclidean":
        return None
    if output_metric not in {
        "next_rmsnorm_gamma",
        "next_rmsnorm_gamma_token_rms",
    }:
        raise ValueError(f"unsupported output metric: {output_metric}")
    layers = model.model.layers
    if not 0 <= layer_index < len(layers):
        raise IndexError("layer_index is outside the decoder layer range")
    if layer_index + 1 < len(layers):
        return layers[layer_index + 1].input_layernorm.weight
    return model.model.norm.weight


def uses_token_rms_metric(output_metric: str) -> bool:
    """Return whether the metric weights tokens by next-normalization RMS."""

    return output_metric == "next_rmsnorm_gamma_token_rms"


def layer_refinement_budget(
    max_swaps: int, layer_index: int, skipped_layers: set[int]
) -> int:
    """Disable local swaps on selected layers while preserving CAMERA masks."""

    return 0 if layer_index in skipped_layers else max_swaps


def pareto_trust_region_swap_budget(retained_channels: int) -> int:
    """Bound Pareto exchanges by the existing one-eighth locality rule."""

    if retained_channels < 0:
        raise ValueError("retained_channels must be non-negative")
    if retained_channels == 0:
        return 0
    return max(1, retained_channels // 8)


def next_rmsnorm_epsilon(model: torch.nn.Module, layer_index: int) -> float:
    """Return the epsilon of the normalization consuming this layer output."""

    layers = model.model.layers
    if not 0 <= layer_index < len(layers):
        raise IndexError("layer_index is outside the decoder layer range")
    norm = (
        layers[layer_index + 1].input_layernorm
        if layer_index + 1 < len(layers)
        else model.model.norm
    )
    return float(norm.variance_epsilon)


def _slice_captured_batch_value(
    value: object, *, batch_size: int, captured_batch_size: int
) -> object:
    if (
        isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == captured_batch_size
    ):
        return value[:batch_size]
    if isinstance(value, tuple):
        return tuple(
            _slice_captured_batch_value(
                item,
                batch_size=batch_size,
                captured_batch_size=captured_batch_size,
            )
            for item in value
        )
    return value


def decoder_layer_hidden(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, object] | LayerReplayContext,
    *,
    layer_index: int = 0,
    batch_start: int = 0,
    replay_pass: str = "default",
    captured_batch_size: int | None = None,
) -> torch.Tensor:
    """Normalize decoder-layer returns across Transformers API versions."""

    args: tuple[object, ...] = ()
    kwargs: dict[str, object]
    if isinstance(layer_kwargs, LayerReplayContext):
        effective_batch_size = (
            hidden_states.shape[0]
            if captured_batch_size is None
            else captured_batch_size
        )
        args, kwargs = layer_kwargs.prepare(
            layer_index,
            batch_size=hidden_states.shape[0],
            captured_batch_size=effective_batch_size,
            batch_start=batch_start,
            replay_pass=replay_pass,
        )
    else:
        kwargs = layer_kwargs
    if captured_batch_size is not None:
        if not 0 < hidden_states.shape[0] <= captured_batch_size:
            raise ValueError("hidden batch exceeds captured_batch_size")
        if not isinstance(layer_kwargs, LayerReplayContext):
            kwargs = {
                key: _slice_captured_batch_value(
                    value,
                    batch_size=hidden_states.shape[0],
                    captured_batch_size=captured_batch_size,
                )
                for key, value in layer_kwargs.items()
            }
    output = layer(hidden_states, *args, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _json_atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model_source = args.model_source or args.model
    source_revision = None if args.model_source else args.revision
    if not torch.cuda.is_available():
        raise RuntimeError("pruning requires one visible CUDA GPU")
    device = torch.device("cuda:0")
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    print(f"loading {args.model}@{args.revision} on CPU", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=source_revision,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    if model.config.moe_intermediate_size != 768:
        raise ValueError("the fixed 50% protocol expects Qwen3 expert width 768")
    if any(
        layer >= len(model.model.layers) for layer in args.skip_refine_layer
    ):
        raise ValueError("--skip-refine-layer is outside the decoder layer range")
    skipped_refine_layers = set(args.skip_refine_layer)
    pareto_domains = args.calibration_aggregation in {
        "pareto_residual_refinement",
        "tail_pareto_residual_refinement",
    }
    trajectory_domains = args.calibration_aggregation in {
        "trajectory_transition_refinement",
        "trajectory_transition_guarded_refinement",
    }
    trajectory_field_domains = args.calibration_aggregation in {
        "tail_capability_pareto_trajectory_field",
        "tail_capability_trajectory_field",
    }
    trajectory_field_guarded = (
        args.calibration_aggregation
        == "tail_capability_pareto_trajectory_field"
    )
    router_neighborhood_domains = (
        args.calibration_aggregation
        == "router_neighborhood_local_separation_refinement"
    )
    loop_orthogonal_domains = (
        args.calibration_aggregation
        == "self_induced_loop_orthogonal_refinement"
    )
    aligned_loop_state_domains = (
        args.calibration_aggregation
        == "causally_aligned_dense_loop_state_restoration"
    )
    loop_refinement_domains = (
        loop_orthogonal_domains or aligned_loop_state_domains
    )
    pareto_base_masks: list[torch.Tensor] | None = None
    if (
        pareto_domains
        or trajectory_domains
        or trajectory_field_domains
        or router_neighborhood_domains
        or loop_refinement_domains
    ):
        if args.pareto_base_checkpoint is None:
            raise RuntimeError("pareto base checkpoint is missing")
        base_checkpoint = torch.load(
            args.pareto_base_checkpoint, map_location="cpu", weights_only=True
        )
        raw_base_masks = base_checkpoint.get("masks")
        if not isinstance(raw_base_masks, list) or len(raw_base_masks) != len(
            model.model.layers
        ):
            raise RuntimeError("pareto base checkpoint has incompatible masks")
        pareto_base_masks = [
            torch.as_tensor(mask, dtype=torch.bool).clone()
            for mask in raw_base_masks
        ]
        if loop_refinement_domains and base_checkpoint.get(
            "calibration_aggregation"
        ) != "universal_termination_capacity_envelope":
            raise RuntimeError(
                "self-induced loop refinement requires an IDEA-058 base checkpoint"
            )

    input_ids, calibration, statistics_token_mask = calibration_tokens(
        model_source,
        source_revision,
        nsamples=args.nsamples,
        sequence_length=args.calib_length,
        seed=args.seed,
        calibration_corpus=args.calibration_corpus,
        local_files_only=args.local_files_only,
        competition_math_source=args.competition_math_source,
        competition_math_exclusion_manifest=(
            args.competition_math_exclusion_manifest
        ),
    )
    loop_token_mask: torch.Tensor | None = None
    gold_token_mask: torch.Tensor | None = None
    loop_contrast_metadata: dict[str, object] | None = None
    loop_onset_metadata: dict[str, object] | None = None
    dense_loop_state_summary: dict[str, object] | None = None
    dense_loop_states: list[torch.Tensor] | None = None
    loop_pair_count = 0
    base_sequence_count = input_ids.shape[0]
    if loop_orthogonal_domains:
        if args.loop_recovery_manifest is None:
            raise RuntimeError("loop recovery manifest is missing")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=source_revision,
            use_fast=False,
            local_files_only=args.local_files_only,
        )
        (
            input_ids,
            statistics_token_mask,
            loop_token_mask,
            gold_token_mask,
            loop_contrast_metadata,
        ) = append_loop_contrast_calibration(
            input_ids,
            statistics_token_mask,
            tokenizer,
            manifest_path=args.loop_recovery_manifest,
            sequence_length=args.calib_length,
        )
        loop_pair_count = int(loop_contrast_metadata["valid_pair_count"])
        calibration["loop_contrast"] = loop_contrast_metadata
    elif aligned_loop_state_domains:
        if args.loop_recovery_manifest is None:
            raise RuntimeError("loop recovery manifest is missing")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=source_revision,
            use_fast=False,
            local_files_only=args.local_files_only,
        )
        (
            input_ids,
            statistics_token_mask,
            loop_token_mask,
            loop_onset_metadata,
        ) = append_loop_onset_calibration(
            input_ids,
            statistics_token_mask,
            tokenizer,
            manifest_path=args.loop_recovery_manifest,
            sequence_length=args.calib_length,
        )
        loop_pair_count = int(loop_onset_metadata["valid_context_count"])
        domain_counts = loop_onset_metadata.get("domain_counts")
        if not isinstance(domain_counts, dict) or any(
            int(domain_counts.get(domain, 0)) <= 0
            for domain in ("reasoning", "hard_math", "code")
        ):
            raise RuntimeError(
                "aligned loop-state contexts must cover reasoning, hard_math, and code"
            )
        calibration["loop_onset"] = loop_onset_metadata
        print("capturing dense aligned loop states", flush=True)
        dense_loop_states, dense_loop_state_summary = capture_dense_loop_states(
            model,
            input_ids[base_sequence_count:],
            loop_token_mask[base_sequence_count:],
            device=device,
            chunk_size=1,
        )
        calibration["dense_loop_state_capture"] = dense_loop_state_summary
    print("capturing first-layer calibration states", flush=True)
    inputs, layer_kwargs = capture_first_layer_inputs(
        model, input_ids, device=device, chunk_size=args.chunk_size
    )
    statistics_token_mask = statistics_token_mask.to(device)
    if loop_token_mask is not None:
        loop_token_mask = loop_token_mask.to(device)
    if gold_token_mask is not None:
        gold_token_mask = gold_token_mask.to(device)
    outputs = torch.empty_like(inputs)
    general_sequence_count = (
        int(calibration["sample_allocation"]["wikitext"])
        if args.calibration_aggregation != "pooled"
        else 0
    )
    phase_domains_requested = args.calibration_aggregation in {
        "anchored_early_tail_phase_domain_robust",
        "anchored_early_tail_phase_capacity_envelope",
        "trajectory_transition_refinement",
        "trajectory_transition_guarded_refinement",
    }
    universal_termination_domains = (
        args.calibration_aggregation
        == "universal_termination_capacity_envelope"
    )
    universal_end_domains = (
        universal_termination_domains or trajectory_field_domains
    )
    termination_token_mask: torch.Tensor | None = None
    response_token_mask: torch.Tensor | None = None
    continuation_token_mask: torch.Tensor | None = None
    if universal_end_domains:
        assistant_ids = calibration.get("assistant_marker_token_ids")
        end_ids = calibration.get("assistant_end_marker_token_ids")
        if (
            not isinstance(assistant_ids, list)
            or not assistant_ids
            or not isinstance(end_ids, list)
            or not end_ids
        ):
            raise RuntimeError(
                "universal termination calibration is missing chat marker ids"
            )
        termination_token_mask = torch.zeros_like(
            statistics_token_mask, dtype=torch.bool
        )
        termination_token_mask[general_sequence_count:] = (
            assistant_chat_end_predictor_token_mask(
                input_ids[general_sequence_count:],
                assistant_marker=torch.tensor(assistant_ids, dtype=torch.long),
                end_marker=torch.tensor(end_ids, dtype=torch.long),
            )
        )
        if trajectory_field_domains:
            response_token_mask = torch.zeros_like(
                statistics_token_mask, dtype=torch.bool
            )
            response_token_mask[general_sequence_count:] = (
                assistant_chat_response_predictor_token_mask(
                    input_ids[general_sequence_count:],
                    assistant_marker=torch.tensor(
                        assistant_ids, dtype=torch.long
                    ),
                    end_marker=torch.tensor(end_ids, dtype=torch.long),
                )
            )
            if bool(
                (
                    termination_token_mask
                    & ~response_token_mask
                ).any().item()
            ):
                raise RuntimeError(
                    "assistant-end support must lie inside response support"
                )
            continuation_token_mask = (
                response_token_mask & ~termination_token_mask
            )
            if not bool(
                continuation_token_mask[general_sequence_count:].any().item()
            ):
                raise RuntimeError(
                    "universal transition field has no continuation targets"
                )
        sample_allocation = calibration["sample_allocation"]
        reasoning_stop = general_sequence_count + int(
            sample_allocation["gsm8k"]
        ) + int(sample_allocation["math_qa"])
        hard_math_stop = reasoning_stop + int(
            sample_allocation["competition_math"]
        )
        for name, domain_mask in (
            (
                "reasoning",
                termination_token_mask[
                    general_sequence_count:reasoning_stop
                ],
            ),
            (
                "hard_math",
                termination_token_mask[reasoning_stop:hard_math_stop],
            ),
            ("code", termination_token_mask[hard_math_stop:]),
        ):
            if not bool(domain_mask.any().item()):
                raise RuntimeError(
                    f"universal termination domain {name} has no end targets"
                )
        termination_token_mask = termination_token_mask.to(device)
        if response_token_mask is not None:
            response_token_mask = response_token_mask.to(device)
        if continuation_token_mask is not None:
            continuation_token_mask = continuation_token_mask.to(device)
    phase_end_token_mask: torch.Tensor | None = None
    phase_continuation_token_mask: torch.Tensor | None = None
    phase_level_sequence_count = 0
    if phase_domains_requested:
        marker_ids = calibration.get("assistant_end_marker_token_ids")
        if not isinstance(marker_ids, list) or not marker_ids:
            raise RuntimeError("phase calibration is missing assistant end marker ids")
        phase_end_token_mask = torch.zeros_like(
            statistics_token_mask, dtype=torch.bool
        )
        phase_end_token_mask[general_sequence_count:] = (
            assistant_end_predictor_token_mask(
                input_ids[general_sequence_count:],
                end_marker=torch.tensor(marker_ids, dtype=torch.long),
                assistant_solution_mask=statistics_token_mask[
                    general_sequence_count:
                ],
            )
        )
        phase_continuation_token_mask = (
            statistics_token_mask & ~phase_end_token_mask
        )
        if bool(
            (
                phase_continuation_token_mask[general_sequence_count:]
                & phase_end_token_mask[general_sequence_count:]
            ).any().item()
        ) or not torch.equal(
            (
                phase_continuation_token_mask[general_sequence_count:]
                | phase_end_token_mask[general_sequence_count:]
            ),
            statistics_token_mask[general_sequence_count:],
        ):
            raise RuntimeError("phase masks are not a disjoint solution partition")
        phase_level_sequence_count = int(
            calibration["sample_allocation"]["competition_math"]
        ) // 2
        if phase_level_sequence_count <= 0 or not bool(
            phase_end_token_mask[general_sequence_count:].any().item()
        ):
            raise RuntimeError("phase calibration has no hard-level end targets")
        phase_end_token_mask = phase_end_token_mask.to(device)
        phase_continuation_token_mask = phase_continuation_token_mask.to(device)
    trajectory_problem_token_mask: torch.Tensor | None = None
    if trajectory_domains:
        if phase_end_token_mask is None:
            raise RuntimeError("trajectory calibration has no end mask")
        (
            trajectory_problem_token_mask,
            phase_continuation_token_mask,
            phase_end_token_mask,
        ) = trajectory_phase_token_masks(
            statistics_token_mask,
            phase_end_token_mask,
            general_sequence_count=general_sequence_count,
        )
    capability_domains = args.calibration_aggregation in {
        "capability_envelope_capacity",
        "capability_complement_capacity",
        "anchored_tail_code_complement_capacity",
        "capability_union",
    }
    math_sequence_count = 0
    if capability_domains:
        sample_allocation = calibration["sample_allocation"]
        if args.calibration_corpus == "competition_math_code_append":
            math_sequence_count = int(sample_allocation["competition_math"])
        else:
            math_sequence_count = int(sample_allocation["gsm8k"]) + int(
                sample_allocation["math_qa"]
            )
    dual_specialist_domains = (
        args.calibration_aggregation
        == "reasoning_early_tail_dual_specialist_capacity"
    )
    universal_capability_domains = args.calibration_aggregation in {
        "universal_capability_envelope",
        "universal_termination_capacity_envelope",
        "tail_capability_pareto_trajectory_field",
        "tail_capability_trajectory_field",
        "router_neighborhood_local_separation_refinement",
        "self_induced_loop_orthogonal_refinement",
        "causally_aligned_dense_loop_state_restoration",
    }
    reasoning_sequence_count = 0
    hard_math_sequence_count = 0
    if dual_specialist_domains or universal_capability_domains:
        sample_allocation = calibration["sample_allocation"]
        reasoning_sequence_count = int(sample_allocation["gsm8k"]) + int(
            sample_allocation["math_qa"]
        )
        hard_math_sequence_count = int(sample_allocation["competition_math"])
    pareto_hard_math_sequence_count = 0
    pareto_reasoning_sequence_count = 0
    if pareto_domains:
        sample_allocation = calibration["sample_allocation"]
        pareto_hard_math_sequence_count = int(
            sample_allocation["competition_math"]
        )
        pareto_reasoning_sequence_count = int(
            sample_allocation["gsm8k"]
        ) + int(sample_allocation["math_qa"])

    masks: list[torch.Tensor] = []
    down_coefficients: list[torch.Tensor] = []
    layer_summaries: list[dict[str, object]] = []
    local_records: list[dict[str, object]] = []
    for layer_index, layer in enumerate(model.model.layers):
        layer_start = time.monotonic()
        layer.to(device)
        moe = layer.mlp
        layer_prune_ratio = layer_budget_prune_ratio(
            args.layer_budget_schedule,
            layer_index,
            len(model.model.layers),
            args.prune_ratio,
        )
        output_metric_scale = next_output_metric_scale(
            model, layer_index, args.output_metric
        )
        effective_calibration_aggregation = layer_capacity_aggregation(
            args.calibration_aggregation,
            layer_index,
            len(model.model.layers),
        )
        effective_statistics_token_mask = layer_statistics_token_mask(
            args.calibration_aggregation,
            layer_index,
            len(model.model.layers),
            statistics_token_mask,
        )
        phase_domains = phase_domains_requested and effective_calibration_aggregation in {
            "phase_domain_robust",
            "phase_capacity_envelope",
        }
        dual_specialist_layer = dual_specialist_domains and (
            effective_calibration_aggregation
            in {"anchored_geometric_capacity", "dual_specialist_capacity"}
        )

        def make_accumulator(seed_offset: int) -> Qwen3LayerStatsAccumulator:
            return Qwen3LayerStatsAccumulator(
                moe,
                reservoir_size=args.tokens_per_expert,
                reservoir_seed=args.seed + layer_index + seed_offset,
                output_metric_scale=output_metric_scale,
                interaction_metric=args.interaction_metric,
                difference_regularization=args.difference_regularization,
                difference_strength=args.difference_strength,
                coefficient_difference_strength=args.coefficient_difference_strength,
            )

        phase_domain_accumulators: tuple[
            Qwen3LayerStatsAccumulator, ...
        ] | None = None
        phase_math_accumulator = None
        level4_accumulator = None
        level5_accumulator = None
        end_accumulator = None
        problem_accumulator = None
        solution_accumulator = None
        response_accumulator = None
        hard_math_accumulator = None
        termination_accumulator = None
        continuation_accumulator = None
        pareto_hard_full_accumulator = None
        pareto_hard_solution_accumulator = None
        if trajectory_domains:
            general_accumulator = make_accumulator(0)
            problem_accumulator = make_accumulator(100_000)
            solution_accumulator = make_accumulator(200_000)
            end_accumulator = make_accumulator(300_000)
            response_accumulator = make_accumulator(400_000)
            reasoning_accumulator = response_accumulator
            math_accumulator = None
            code_accumulator = None
            accumulator = DomainRobustStatsAccumulator(
                general_accumulator,
                response_accumulator,
                aggregation="domain_robust",
            )
        elif pareto_domains:
            general_accumulator = make_accumulator(0)
            pareto_hard_full_accumulator = make_accumulator(100_000)
            pareto_hard_solution_accumulator = make_accumulator(200_000)
            reasoning_accumulator = make_accumulator(300_000)
            math_accumulator = None
            code_accumulator = None
            accumulator = general_accumulator
        elif phase_domains:
            general_accumulator = make_accumulator(0)
            phase_capacity_envelope = (
                effective_calibration_aggregation == "phase_capacity_envelope"
            )
            phase_math_accumulator = (
                make_accumulator(100_000) if phase_capacity_envelope else None
            )
            phase_offset = 100_000 if phase_capacity_envelope else 0
            level4_accumulator = make_accumulator(100_000 + phase_offset)
            level5_accumulator = make_accumulator(200_000 + phase_offset)
            end_accumulator = make_accumulator(300_000 + phase_offset)
            phase_domain_accumulators = (
                general_accumulator,
                level4_accumulator,
                level5_accumulator,
                end_accumulator,
            )
            reasoning_accumulator = None
            math_accumulator = None
            code_accumulator = None
            accumulator = (
                PhaseCapacityEnvelopeStatsAccumulator(
                    general_accumulator,
                    phase_math_accumulator,
                    level4_accumulator,
                    level5_accumulator,
                    end_accumulator,
                )
                if phase_math_accumulator is not None
                else MultiDomainRobustStatsAccumulator(
                    *phase_domain_accumulators
                )
            )
        elif universal_capability_domains:
            general_accumulator = make_accumulator(0)
            reasoning_accumulator = make_accumulator(100_000)
            hard_math_accumulator = make_accumulator(200_000)
            code_accumulator = make_accumulator(300_000)
            termination_accumulator = (
                make_accumulator(400_000)
                if universal_end_domains
                else None
            )
            continuation_accumulator = (
                make_accumulator(500_000)
                if trajectory_field_domains
                else None
            )
            math_accumulator = None
            accumulator = (
                UniversalTerminationCapacityEnvelopeStatsAccumulator(
                    general_accumulator,
                    reasoning_accumulator,
                    hard_math_accumulator,
                    code_accumulator,
                    termination_accumulator,
                )
                if universal_termination_domains
                else UniversalCapabilityEnvelopeStatsAccumulator(
                    general_accumulator,
                    reasoning_accumulator,
                    hard_math_accumulator,
                    code_accumulator,
                )
            )
        elif dual_specialist_layer:
            general_accumulator = make_accumulator(0)
            reasoning_accumulator = make_accumulator(100_000)
            math_accumulator = None
            code_accumulator = None
            if effective_calibration_aggregation == "dual_specialist_capacity":
                hard_math_accumulator = make_accumulator(200_000)
                code_accumulator = make_accumulator(300_000)
                accumulator = DualSpecialistCapacityEnvelopeStatsAccumulator(
                    general_accumulator,
                    reasoning_accumulator,
                    hard_math_accumulator,
                    code_accumulator,
                )
            else:
                accumulator = DomainRobustStatsAccumulator(
                    general_accumulator,
                    reasoning_accumulator,
                    aggregation=effective_calibration_aggregation,
                )
        elif capability_domains:
            general_accumulator = make_accumulator(0)
            math_accumulator = make_accumulator(100_000)
            code_accumulator = make_accumulator(200_000)
            reasoning_accumulator = None
            if effective_calibration_aggregation == "anchored_geometric_capacity":
                accumulator = DomainRobustStatsAccumulator(
                    general_accumulator,
                    math_accumulator,
                    aggregation=effective_calibration_aggregation,
                )
            else:
                accumulator_classes = {
                    "capability_envelope_capacity": CapabilityEnvelopeStatsAccumulator,
                    "capability_complement_capacity": CapabilityComplementStatsAccumulator,
                    "capability_union": CapabilityUnionStatsAccumulator,
                }
                accumulator_class = accumulator_classes[
                    effective_calibration_aggregation
                ]
                accumulator = accumulator_class(
                    general_accumulator, math_accumulator, code_accumulator
                )
        elif args.calibration_aggregation != "pooled":
            general_accumulator = make_accumulator(0)
            reasoning_accumulator = make_accumulator(100_000)
            math_accumulator = None
            code_accumulator = None
            accumulator = DomainRobustStatsAccumulator(
                general_accumulator,
                reasoning_accumulator,
                aggregation=effective_calibration_aggregation,
            )
        else:
            general_accumulator = None
            reasoning_accumulator = None
            math_accumulator = None
            code_accumulator = None
            accumulator = make_accumulator(0)

        loop_phase_means: torch.Tensor | None = None
        gold_phase_means: torch.Tensor | None = None
        loop_phase_routed_counts: torch.Tensor | None = None
        gold_phase_routed_counts: torch.Tensor | None = None
        loop_phase_selected_counts: torch.Tensor | None = None
        gold_phase_selected_counts: torch.Tensor | None = None
        if loop_refinement_domains:
            if loop_pair_count <= 0:
                raise RuntimeError("self-induced loop calibration has no pairs")
            phase_shape = (
                loop_pair_count,
                accumulator.expert_count,
                accumulator.channel_count,
            )
            routed_shape = (loop_pair_count, accumulator.expert_count)
            loop_phase_means = torch.zeros(
                phase_shape, dtype=torch.float32, device=device
            )
            loop_phase_routed_counts = torch.zeros(
                routed_shape, dtype=torch.long, device=device
            )
            loop_phase_selected_counts = torch.zeros(
                loop_pair_count, dtype=torch.long, device=device
            )
            if loop_orthogonal_domains:
                gold_phase_means = torch.zeros_like(loop_phase_means)
                gold_phase_routed_counts = torch.zeros_like(
                    loop_phase_routed_counts
                )
                gold_phase_selected_counts = torch.zeros_like(
                    loop_phase_selected_counts
                )
            else:
                if dense_loop_states is None:
                    raise RuntimeError("dense aligned loop states are missing")
                dense_layer_states = dense_loop_states[layer_index]
                if dense_layer_states.shape != phase_shape:
                    raise RuntimeError(
                        "dense aligned loop-state layer has incompatible shape"
                    )
                gold_phase_means = dense_layer_states.to(
                    device=device, dtype=torch.float32
                )
                gold_phase_routed_counts = (
                    gold_phase_means.abs().sum(dim=2) > 0
                ).to(torch.long)
                gold_phase_selected_counts = torch.ones(
                    loop_pair_count, dtype=torch.long, device=device
                )

        token_rms_metric = uses_token_rms_metric(args.output_metric)
        pending_moe_input: dict[str, torch.Tensor] = {}
        capture_for_manual_update = (
            token_rms_metric
            or args.calibration_aggregation != "pooled"
            or not bool(effective_statistics_token_mask.all().item())
        )
        if capture_for_manual_update:
            def capture_moe_input(
                _module: torch.nn.Module, hook_inputs: tuple[torch.Tensor, ...]
            ) -> None:
                if pending_moe_input:
                    raise RuntimeError("MoE input capture was not consumed")
                pending_moe_input["hidden_states"] = hook_inputs[0]

            handle = moe.register_forward_pre_hook(capture_moe_input)
        else:
            handle = moe.register_forward_pre_hook(make_stats_hook(accumulator))
        for start in range(0, inputs.shape[0], args.chunk_size):
            stop = start + args.chunk_size
            outputs[start:stop] = decoder_layer_hidden(
                layer,
                inputs[start:stop],
                layer_kwargs,
                layer_index=layer_index,
                batch_start=start,
                replay_pass="statistics",
                captured_batch_size=min(args.chunk_size, inputs.shape[0]),
            )
            if capture_for_manual_update:
                moe_inputs = pending_moe_input.pop("hidden_states", None)
                if moe_inputs is None or pending_moe_input:
                    raise RuntimeError("failed to capture exactly one MoE input")
                token_metric_scale = None
                if token_rms_metric:
                    next_residual = outputs[start:stop].float()
                    token_metric_scale = next_residual.square().mean(dim=-1)
                    token_metric_scale = token_metric_scale.add(
                        next_rmsnorm_epsilon(model, layer_index)
                    ).reciprocal()
                if trajectory_domains:
                    if any(
                        domain is None
                        for domain in (
                            general_accumulator,
                            problem_accumulator,
                            solution_accumulator,
                            end_accumulator,
                            response_accumulator,
                        )
                    ) or any(
                        mask is None
                        for mask in (
                            trajectory_problem_token_mask,
                            phase_continuation_token_mask,
                            phase_end_token_mask,
                        )
                    ):
                        raise RuntimeError("missing trajectory-domain inputs")
                    general_stop = domain_general_prefix_count(
                        start, stop, general_sequence_count
                    )
                    if general_stop:
                        general_accumulator.update(
                            moe_inputs[:general_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[:general_stop]
                            ),
                            token_selection_mask=torch.ones_like(
                                statistics_token_mask[start : start + general_stop]
                            ),
                        )
                    if general_stop < stop - start:
                        math_inputs = moe_inputs[general_stop:]
                        math_metric = (
                            None
                            if token_metric_scale is None
                            else token_metric_scale[general_stop:]
                        )
                        math_start = start + general_stop
                        for domain_accumulator, selection_mask in (
                            (problem_accumulator, trajectory_problem_token_mask),
                            (solution_accumulator, phase_continuation_token_mask),
                            (end_accumulator, phase_end_token_mask),
                            (response_accumulator, statistics_token_mask),
                        ):
                            domain_accumulator.update(
                                math_inputs,
                                token_metric_scale=math_metric,
                                token_selection_mask=selection_mask[
                                    math_start:stop
                                ],
                            )
                elif pareto_domains:
                    if any(
                        domain is None
                        for domain in (
                            general_accumulator,
                            pareto_hard_full_accumulator,
                            pareto_hard_solution_accumulator,
                            reasoning_accumulator,
                        )
                    ):
                        raise RuntimeError("missing pareto-domain accumulators")
                    general_count, hard_count, reasoning_count = (
                        pareto_domain_batch_counts(
                            start,
                            stop,
                            general_sequence_count,
                            pareto_hard_math_sequence_count,
                        )
                    )
                    general_stop = general_count
                    hard_stop = general_stop + hard_count
                    reasoning_stop = hard_stop + reasoning_count
                    if general_count:
                        general_accumulator.update(
                            moe_inputs[:general_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[:general_stop]
                            ),
                            token_selection_mask=statistics_token_mask[
                                start : start + general_stop
                            ],
                        )
                    if hard_count:
                        hard_metric = (
                            None
                            if token_metric_scale is None
                            else token_metric_scale[general_stop:hard_stop]
                        )
                        hard_inputs = moe_inputs[general_stop:hard_stop]
                        pareto_hard_full_accumulator.update(
                            hard_inputs,
                            token_metric_scale=hard_metric,
                            token_selection_mask=torch.ones_like(
                                statistics_token_mask[
                                    start + general_stop : start + hard_stop
                                ],
                                dtype=torch.bool,
                            ),
                        )
                        pareto_hard_solution_accumulator.update(
                            hard_inputs,
                            token_metric_scale=hard_metric,
                            token_selection_mask=statistics_token_mask[
                                start + general_stop : start + hard_stop
                            ],
                        )
                    if reasoning_count:
                        reasoning_accumulator.update(
                            moe_inputs[hard_stop:reasoning_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[
                                    hard_stop:reasoning_stop
                                ]
                            ),
                            token_selection_mask=statistics_token_mask[
                                start + hard_stop : start + reasoning_stop
                            ],
                        )
                elif phase_domains:
                    if (
                        general_accumulator is None
                        or level4_accumulator is None
                        or level5_accumulator is None
                        or end_accumulator is None
                        or phase_continuation_token_mask is None
                        or phase_end_token_mask is None
                    ):
                        raise RuntimeError("missing phase calibration accumulators")

                    def update_phase_domain(
                        domain_accumulator: Qwen3LayerStatsAccumulator,
                        domain_start: int,
                        domain_stop: int,
                        selection_mask: torch.Tensor,
                    ) -> None:
                        selected_start = max(start, domain_start)
                        selected_stop = min(stop, domain_stop)
                        if selected_start >= selected_stop:
                            return
                        batch_start = selected_start - start
                        batch_stop = selected_stop - start
                        domain_accumulator.update(
                            moe_inputs[batch_start:batch_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[batch_start:batch_stop]
                            ),
                            token_selection_mask=selection_mask[
                                selected_start:selected_stop
                            ],
                        )

                    level4_start = general_sequence_count
                    level5_start = level4_start + phase_level_sequence_count
                    math_stop = inputs.shape[0]
                    update_phase_domain(
                        general_accumulator,
                        0,
                        general_sequence_count,
                        effective_statistics_token_mask,
                    )
                    if phase_math_accumulator is not None:
                        update_phase_domain(
                            phase_math_accumulator,
                            level4_start,
                            math_stop,
                            torch.ones_like(statistics_token_mask),
                        )
                    update_phase_domain(
                        level4_accumulator,
                        level4_start,
                        level5_start,
                        phase_continuation_token_mask,
                    )
                    update_phase_domain(
                        level5_accumulator,
                        level5_start,
                        math_stop,
                        phase_continuation_token_mask,
                    )
                    update_phase_domain(
                        end_accumulator,
                        level4_start,
                        math_stop,
                        phase_end_token_mask,
                    )
                elif loop_refinement_domains:
                    if any(
                        value is None
                        for value in (
                            general_accumulator,
                            reasoning_accumulator,
                            hard_math_accumulator,
                            code_accumulator,
                            loop_token_mask,
                            loop_phase_means,
                            gold_phase_means,
                            loop_phase_routed_counts,
                            gold_phase_routed_counts,
                            loop_phase_selected_counts,
                            gold_phase_selected_counts,
                        )
                    ):
                        raise RuntimeError(
                            "missing self-induced loop refinement inputs"
                        )
                    if loop_orthogonal_domains and gold_token_mask is None:
                        raise RuntimeError("loop contrast gold mask is missing")
                    base_stop = min(stop, base_sequence_count)
                    if start < base_stop:
                        domain_counts = specialist_domain_batch_counts(
                            start,
                            base_stop,
                            general_sequence_count,
                            reasoning_sequence_count,
                            hard_math_sequence_count,
                        )
                        domain_accumulators = (
                            general_accumulator,
                            reasoning_accumulator,
                            hard_math_accumulator,
                            code_accumulator,
                        )
                        domain_start = 0
                        for domain_accumulator, domain_count in zip(
                            domain_accumulators, domain_counts
                        ):
                            domain_stop = domain_start + domain_count
                            if domain_count:
                                domain_accumulator.update(
                                    moe_inputs[domain_start:domain_stop],
                                    token_metric_scale=(
                                        None
                                        if token_metric_scale is None
                                        else token_metric_scale[
                                            domain_start:domain_stop
                                        ]
                                    ),
                                    token_selection_mask=(
                                        effective_statistics_token_mask[
                                            start + domain_start : start + domain_stop
                                        ]
                                    ),
                                )
                            domain_start = domain_stop

                    def update_loop_phase(
                        phase_start: int,
                        phase_stop: int,
                        selection_mask: torch.Tensor,
                        phase_means: torch.Tensor,
                        phase_routed_counts: torch.Tensor,
                        phase_selected_counts: torch.Tensor,
                    ) -> None:
                        selected_start = max(start, phase_start)
                        selected_stop = min(stop, phase_stop)
                        if selected_start >= selected_stop:
                            return
                        batch_start = selected_start - start
                        batch_stop = selected_stop - start
                        means, routed, selected = router_weighted_phase_means(
                            moe,
                            moe_inputs[batch_start:batch_stop],
                            token_selection_mask=selection_mask[
                                selected_start:selected_stop
                            ],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[batch_start:batch_stop]
                            ),
                        )
                        pair_start = selected_start - phase_start
                        pair_stop = selected_stop - phase_start
                        phase_means[pair_start:pair_stop] = means
                        phase_routed_counts[pair_start:pair_stop] = routed
                        phase_selected_counts[pair_start:pair_stop] = selected

                    loop_start = base_sequence_count
                    loop_stop = loop_start + loop_pair_count
                    gold_start = loop_stop
                    gold_stop = gold_start + loop_pair_count
                    update_loop_phase(
                        loop_start,
                        loop_stop,
                        loop_token_mask,
                        loop_phase_means,
                        loop_phase_routed_counts,
                        loop_phase_selected_counts,
                    )
                    if loop_orthogonal_domains:
                        update_loop_phase(
                            gold_start,
                            gold_stop,
                            gold_token_mask,
                            gold_phase_means,
                            gold_phase_routed_counts,
                            gold_phase_selected_counts,
                        )
                elif universal_capability_domains or dual_specialist_layer:
                    if general_accumulator is None or reasoning_accumulator is None:
                        raise RuntimeError(
                            "missing four-domain capability accumulators"
                        )
                    domain_counts = specialist_domain_batch_counts(
                        start,
                        stop,
                        general_sequence_count,
                        reasoning_sequence_count,
                        hard_math_sequence_count,
                    )
                    domain_accumulators = (
                        general_accumulator,
                        reasoning_accumulator,
                        hard_math_accumulator,
                        code_accumulator,
                    )
                    domain_start = 0
                    for domain_accumulator, domain_count in zip(
                        domain_accumulators, domain_counts
                    ):
                        domain_stop = domain_start + domain_count
                        if domain_accumulator is not None and domain_count:
                            domain_accumulator.update(
                                moe_inputs[domain_start:domain_stop],
                                token_metric_scale=(
                                    None
                                    if token_metric_scale is None
                                    else token_metric_scale[
                                        domain_start:domain_stop
                                    ]
                                ),
                                token_selection_mask=(
                                    effective_statistics_token_mask[
                                        start + domain_start : start + domain_stop
                                    ]
                                ),
                            )
                        domain_start = domain_stop
                    if universal_end_domains:
                        if (
                            termination_accumulator is None
                            or termination_token_mask is None
                        ):
                            raise RuntimeError(
                                "missing universal termination inputs"
                            )
                        specialist_start = domain_counts[0]
                        if specialist_start < stop - start:
                            termination_accumulator.update(
                                moe_inputs[specialist_start:],
                                token_metric_scale=(
                                    None
                                    if token_metric_scale is None
                                    else token_metric_scale[specialist_start:]
                                ),
                                token_selection_mask=termination_token_mask[
                                    start + specialist_start : stop
                                ],
                            )
                            if trajectory_field_domains:
                                if (
                                    continuation_accumulator is None
                                    or continuation_token_mask is None
                                ):
                                    raise RuntimeError(
                                        "missing universal continuation inputs"
                                    )
                                continuation_accumulator.update(
                                    moe_inputs[specialist_start:],
                                    token_metric_scale=(
                                        None
                                        if token_metric_scale is None
                                        else token_metric_scale[
                                            specialist_start:
                                        ]
                                    ),
                                    token_selection_mask=(
                                        continuation_token_mask[
                                            start + specialist_start : stop
                                        ]
                                    ),
                                )
                elif capability_domains:
                    if (
                        general_accumulator is None
                        or math_accumulator is None
                        or code_accumulator is None
                    ):
                        raise RuntimeError(
                            "missing capability calibration accumulators"
                        )
                    general_count, math_count, code_count = (
                        capability_domain_batch_counts(
                            start,
                            stop,
                            general_sequence_count,
                            math_sequence_count,
                        )
                    )
                    domain_slices = (
                        (general_accumulator, 0, general_count),
                        (
                            math_accumulator,
                            general_count,
                            general_count + math_count,
                        ),
                        (
                            code_accumulator,
                            general_count + math_count,
                            general_count + math_count + code_count,
                        ),
                    )
                    for domain_accumulator, domain_start, domain_stop in domain_slices:
                        if domain_start == domain_stop:
                            continue
                        domain_accumulator.update(
                            moe_inputs[domain_start:domain_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[domain_start:domain_stop]
                            ),
                            token_selection_mask=effective_statistics_token_mask[
                                start + domain_start : start + domain_stop
                            ],
                        )
                elif args.calibration_aggregation != "pooled":
                    if general_accumulator is None or reasoning_accumulator is None:
                        raise RuntimeError("missing domain calibration accumulators")
                    general_stop = domain_general_prefix_count(
                        start,
                        stop,
                        general_sequence_count,
                    )
                    if general_stop:
                        general_accumulator.update(
                            moe_inputs[:general_stop],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[:general_stop]
                            ),
                            token_selection_mask=effective_statistics_token_mask[
                                start : start + general_stop
                            ],
                        )
                    if general_stop < stop - start:
                        reasoning_accumulator.update(
                            moe_inputs[general_stop:],
                            token_metric_scale=(
                                None
                                if token_metric_scale is None
                                else token_metric_scale[general_stop:]
                            ),
                            token_selection_mask=effective_statistics_token_mask[
                                start + general_stop : stop
                            ],
                        )
                else:
                    accumulator.update(
                        moe_inputs,
                        token_metric_scale=token_metric_scale,
                        token_selection_mask=effective_statistics_token_mask[
                            start:stop
                        ],
                    )
        handle.remove()

        phase_domain_summary: dict[str, dict[str, int]] | None = None
        if phase_domains:
            if (
                phase_domain_accumulators is None
                or phase_continuation_token_mask is None
                or phase_end_token_mask is None
            ):
                raise RuntimeError("phase-domain summary inputs are missing")
            level4_start = general_sequence_count
            level5_start = level4_start + phase_level_sequence_count
            phase_specs = (
                (
                    "wikitext_all",
                    phase_domain_accumulators[0],
                    effective_statistics_token_mask[:general_sequence_count],
                ),
                (
                    "level4_continuation",
                    phase_domain_accumulators[1],
                    phase_continuation_token_mask[
                        level4_start:level5_start
                    ],
                ),
                (
                    "level5_continuation",
                    phase_domain_accumulators[2],
                    phase_continuation_token_mask[level5_start:],
                ),
                (
                    "assistant_end",
                    phase_domain_accumulators[3],
                    phase_end_token_mask[level4_start:],
                ),
            )
            if phase_math_accumulator is not None:
                phase_specs = (
                    phase_specs[0],
                    (
                        "hardmath_all",
                        phase_math_accumulator,
                        torch.ones_like(
                            statistics_token_mask[level4_start:],
                            dtype=torch.bool,
                        ),
                    ),
                    *phase_specs[1:],
                )
            phase_domain_summary = {}
            for name, domain_accumulator, domain_mask in phase_specs:
                routed = domain_accumulator.routed_token_counts
                statistics_tokens = int(domain_mask.sum().item())
                if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                    raise RuntimeError(f"phase domain {name} has no observations")
                phase_domain_summary[name] = {
                    "statistics_tokens": statistics_tokens,
                    "routed_tokens": int(routed.sum().item()),
                    "min_expert_routed_tokens": int(routed.min().item()),
                    "max_expert_routed_tokens": int(routed.max().item()),
                }
            phase_specs = ()

        specialist_domain_summary: dict[str, dict[str, int]] | None = None
        if universal_capability_domains or (
            dual_specialist_layer
            and effective_calibration_aggregation == "dual_specialist_capacity"
        ):
            if any(
                domain is None
                for domain in (
                    general_accumulator,
                    reasoning_accumulator,
                    hard_math_accumulator,
                    code_accumulator,
                )
            ):
                raise RuntimeError("dual-specialist summary inputs are missing")
            specialist_specs = (
                ("wikitext_all", general_accumulator, general_sequence_count),
                (
                    "reasoning_all",
                    reasoning_accumulator,
                    reasoning_sequence_count,
                ),
                (
                    "hardmath_all",
                    hard_math_accumulator,
                    hard_math_sequence_count,
                ),
                (
                    "mbpp_all",
                    code_accumulator,
                    (base_sequence_count if loop_refinement_domains else inputs.shape[0])
                    - general_sequence_count
                    - reasoning_sequence_count
                    - hard_math_sequence_count,
                ),
            )
            specialist_domain_summary = {}
            for name, domain_accumulator, sequence_count in specialist_specs:
                if domain_accumulator is None:
                    raise RuntimeError(f"specialist domain {name} is missing")
                routed = domain_accumulator.routed_token_counts
                statistics_tokens = sequence_count * args.calib_length
                if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                    raise RuntimeError(
                        f"specialist domain {name} has no observations"
                    )
                specialist_domain_summary[name] = {
                    "statistics_tokens": statistics_tokens,
                    "routed_tokens": int(routed.sum().item()),
                    "min_expert_routed_tokens": int(routed.min().item()),
                    "max_expert_routed_tokens": int(routed.max().item()),
                }
            if universal_end_domains:
                if (
                    termination_accumulator is None
                    or termination_token_mask is None
                ):
                    raise RuntimeError(
                        "universal termination summary inputs are missing"
                    )
                routed = termination_accumulator.routed_token_counts
                statistics_tokens = int(termination_token_mask.sum().item())
                if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                    raise RuntimeError(
                        "universal assistant-end domain has no observations"
                    )
                specialist_domain_summary["assistant_end"] = {
                    "statistics_tokens": statistics_tokens,
                    "routed_tokens": int(routed.sum().item()),
                    "min_expert_routed_tokens": int(routed.min().item()),
                    "max_expert_routed_tokens": int(routed.max().item()),
                }
                if trajectory_field_domains:
                    if (
                        continuation_accumulator is None
                        or continuation_token_mask is None
                    ):
                        raise RuntimeError(
                            "universal continuation summary inputs are missing"
                        )
                    routed = continuation_accumulator.routed_token_counts
                    statistics_tokens = int(
                        continuation_token_mask.sum().item()
                    )
                    if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                        raise RuntimeError(
                            "universal continuation domain has no observations"
                        )
                    specialist_domain_summary["assistant_continuation"] = {
                        "statistics_tokens": statistics_tokens,
                        "routed_tokens": int(routed.sum().item()),
                        "min_expert_routed_tokens": int(routed.min().item()),
                        "max_expert_routed_tokens": int(routed.max().item()),
                    }

        pareto_domain_summary: dict[str, dict[str, int]] | None = None
        trajectory_domain_summary: dict[str, dict[str, int]] | None = None
        trajectory_field_trace: dict[str, object] | None = None
        router_neighborhood_summary: dict[str, object] | None = None
        loop_orthogonal_summary: dict[str, object] | None = None
        propagation_mask: torch.Tensor | None = None
        if loop_refinement_domains:
            if (
                pareto_base_masks is None
                or general_accumulator is None
                or loop_phase_means is None
                or gold_phase_means is None
                or loop_phase_routed_counts is None
                or gold_phase_routed_counts is None
                or loop_phase_selected_counts is None
                or gold_phase_selected_counts is None
            ):
                raise RuntimeError(
                    "self-induced loop refinement inputs are missing"
                )
            if (
                (loop_phase_selected_counts <= 0).any()
                or (gold_phase_selected_counts <= 0).any()
            ):
                raise RuntimeError(
                    "every loop contrast pair must contain both causal phases"
                )
            base_mask = pareto_base_masks[layer_index].to(device)
            if base_mask.shape != (
                accumulator.expert_count,
                accumulator.channel_count,
            ):
                raise RuntimeError(
                    "self-induced loop base mask has incompatible shape"
                )
            expected_layer_kept = round(
                base_mask.numel() * (1 - layer_prune_ratio)
            )
            if int(base_mask.sum().item()) != expected_layer_kept:
                raise RuntimeError(
                    "self-induced loop base mask violates layer budget"
                )
            if aligned_loop_state_domains:
                loop_phase_means = (
                    loop_phase_means * base_mask.unsqueeze(0).to(torch.float32)
                )
            anchor_importance = accumulator.camera_importance(
                args.camera_alpha
            )
            keep_mask = torch.zeros_like(base_mask)
            records = []
            fallback_experts = 0
            recovery_experts = 0
            for expert_index in range(accumulator.expert_count):
                initial = torch.nonzero(
                    base_mask[expert_index], as_tuple=False
                ).flatten()
                recovery_importance, recovery_trace = (
                    build_loop_orthogonal_importance(
                        loop_phase_means[:, expert_index],
                        gold_phase_means[:, expert_index],
                        general_accumulator.metric_down_projection(
                            expert_index
                        ),
                        anchor_importance=anchor_importance[expert_index],
                        return_trace=True,
                    )
                )
                fallback = bool(recovery_trace["fallback"])
                fallback_experts += int(fallback)
                recovery_experts += int(not fallback)
                objective = reweight_gram_with_importance(
                    accumulator.interaction_gram(expert_index),
                    recovery_importance,
                    interaction_strength=args.interaction_strength,
                )
                if fallback or initial.numel() in {
                    0,
                    accumulator.channel_count,
                }:
                    refined = initial
                else:
                    refined = swap_refine_keep_indices(
                        objective, initial, max_swaps=1
                    )
                keep_mask[expert_index, refined] = True
                swap_count = int(
                    (
                        keep_mask[expert_index]
                        != base_mask[expert_index]
                    ).sum().item()
                    // 2
                )
                initial_error = gram_reconstruction_error(
                    objective, initial
                )
                refined_error = gram_reconstruction_error(
                    objective, refined
                )
                error_delta = refined_error - initial_error
                error_tolerance = (
                    torch.finfo(torch.float32).eps
                    * initial_error.abs().clamp_min(1)
                )
                if swap_count > 1:
                    raise RuntimeError(
                        "self-induced loop expert exceeded its single-swap budget"
                    )
                if swap_count > 0 and error_delta >= -error_tolerance:
                    raise RuntimeError(
                        "self-induced loop swap did not strictly improve its objective"
                    )
                records.append(
                    {
                        "expert": expert_index,
                        "retained_channels": int(initial.numel()),
                        "selection_status": (
                            (
                                "dense_loop_state_refined"
                                if aligned_loop_state_domains
                                else "loop_orthogonal_refined"
                            )
                            if swap_count
                            else (
                                (
                                    "dense_loop_state_fallback"
                                    if aligned_loop_state_domains
                                    else "loop_orthogonal_fallback"
                                )
                                if fallback
                                else (
                                    "dense_loop_state_fixed_point"
                                    if aligned_loop_state_domains
                                    else "loop_orthogonal_fixed_point"
                                )
                            )
                        ),
                        "swap_budget": (
                            0
                            if fallback
                            or initial.numel()
                            in {0, accumulator.channel_count}
                            else 1
                        ),
                        "swap_count": swap_count,
                        "initial_objective_error": float(
                            initial_error.item()
                        ),
                        "refined_objective_error": float(
                            refined_error.item()
                        ),
                        "objective_error_delta": float(error_delta.item()),
                        "objective_error_tolerance": float(
                            error_tolerance.item()
                        ),
                        "loop_routed_tokens": int(
                            loop_phase_routed_counts[
                                :, expert_index
                            ].sum().item()
                        ),
                        "gold_routed_tokens": int(
                            gold_phase_routed_counts[
                                :, expert_index
                            ].sum().item()
                        ),
                        "recovery": recovery_trace,
                    }
                )
            if not torch.equal(
                keep_mask.sum(dim=1), base_mask.sum(dim=1)
            ):
                raise RuntimeError(
                    "self-induced loop refinement changed expert widths"
                )
            if recovery_experts <= 0:
                raise RuntimeError(
                    "self-induced loop refinement has no recovery evidence"
                )
            propagation_mask = base_mask
            layer_coefficients = None
            loop_orthogonal_summary = {
                "pair_count": loop_pair_count,
                "state_source": (
                    "aligned_dense_same_prefix"
                    if aligned_loop_state_domains
                    else "unaligned_train_gold_branch"
                ),
                "loop_statistics_tokens": int(
                    loop_phase_selected_counts.sum().item()
                ),
                "gold_statistics_tokens": int(
                    gold_phase_selected_counts.sum().item()
                ),
                "loop_routed_tokens": int(
                    loop_phase_routed_counts.sum().item()
                ),
                "gold_routed_tokens": int(
                    gold_phase_routed_counts.sum().item()
                ),
                "recovery_experts": recovery_experts,
                "fallback_experts": fallback_experts,
                "changed_experts": sum(
                    int(record["swap_count"] > 0) for record in records
                ),
                "propagation": "base_checkpoint",
            }
        elif router_neighborhood_domains:
            if pareto_base_masks is None or any(
                domain is None
                for domain in (
                    general_accumulator,
                    reasoning_accumulator,
                    hard_math_accumulator,
                    code_accumulator,
                )
            ):
                raise RuntimeError(
                    "router-neighborhood refinement inputs are missing"
                )
            base_mask = pareto_base_masks[layer_index].to(device)
            if base_mask.shape != (
                accumulator.expert_count,
                accumulator.channel_count,
            ):
                raise RuntimeError(
                    "router-neighborhood base mask has incompatible shape"
                )
            expected_layer_kept = round(
                base_mask.numel() * (1 - layer_prune_ratio)
            )
            if int(base_mask.sum().item()) != expected_layer_kept:
                raise RuntimeError(
                    "router-neighborhood base mask violates layer budget"
                )
            domain_names = (
                "wikitext_all",
                "reasoning_all",
                "hardmath_all",
                "mbpp_all",
            )
            domain_accumulators = (
                general_accumulator,
                reasoning_accumulator,
                hard_math_accumulator,
                code_accumulator,
            )
            robust_importance = accumulator.camera_importance(
                args.camera_alpha
            )
            keep_mask = torch.zeros_like(base_mask)
            records = []
            pair_totals = {name: 0 for name in domain_names}
            fallback_totals = {name: 0 for name in domain_names}
            for expert_index in range(accumulator.expert_count):
                initial = torch.nonzero(
                    base_mask[expert_index], as_tuple=False
                ).flatten()
                ordinary_grams = tuple(
                    domain.interaction_gram(expert_index)
                    for domain in domain_accumulators
                )
                local_with_trace = tuple(
                    domain.local_separation_gram(
                        expert_index, return_trace=True
                    )
                    for domain in domain_accumulators
                )
                local_grams = tuple(value[0] for value in local_with_trace)
                local_traces = tuple(value[1] for value in local_with_trace)
                objective = universal_router_neighborhood_objective(
                    ordinary_grams, local_grams
                )
                objective = reweight_gram_with_importance(
                    objective,
                    robust_importance[expert_index],
                    interaction_strength=args.interaction_strength,
                )
                if initial.numel() in {0, accumulator.channel_count}:
                    refined = initial
                else:
                    refined = swap_refine_keep_indices(
                        objective, initial, max_swaps=1
                    )
                keep_mask[expert_index, refined] = True
                swap_count = int(
                    (
                        keep_mask[expert_index]
                        != base_mask[expert_index]
                    ).sum().item()
                    // 2
                )
                initial_error = gram_reconstruction_error(
                    objective, initial
                )
                refined_error = gram_reconstruction_error(
                    objective, refined
                )
                error_delta = refined_error - initial_error
                error_tolerance = (
                    torch.finfo(torch.float32).eps
                    * initial_error.abs().clamp_min(1)
                )
                if swap_count > 0 and error_delta >= -error_tolerance:
                    raise RuntimeError(
                        "router-neighborhood swap did not strictly improve "
                        "its objective"
                    )
                domain_pair_counts = {}
                domain_fallbacks = {}
                domain_fallback_reasons = {}
                for name, trace in zip(domain_names, local_traces):
                    pair_count = int(trace["pair_count"])
                    fallback = bool(trace["fallback"])
                    pair_totals[name] += pair_count
                    fallback_totals[name] += int(fallback)
                    domain_pair_counts[name] = pair_count
                    domain_fallbacks[name] = fallback
                    domain_fallback_reasons[name] = trace[
                        "fallback_reason"
                    ]
                records.append(
                    {
                        "expert": expert_index,
                        "retained_channels": int(initial.numel()),
                        "selection_status": (
                            "router_neighborhood_refined"
                            if swap_count
                            else "router_neighborhood_fixed_point"
                        ),
                        "swap_budget": (
                            1
                            if initial.numel()
                            not in {0, accumulator.channel_count}
                            else 0
                        ),
                        "swap_count": swap_count,
                        "initial_objective_error": float(
                            initial_error.item()
                        ),
                        "refined_objective_error": float(
                            refined_error.item()
                        ),
                        "objective_error_delta": float(error_delta.item()),
                        "objective_error_tolerance": float(
                            error_tolerance.item()
                        ),
                        "domain_pair_counts": domain_pair_counts,
                        "domain_fallbacks": domain_fallbacks,
                        "domain_fallback_reasons": domain_fallback_reasons,
                    }
                )
            if not torch.equal(
                keep_mask.sum(dim=1), base_mask.sum(dim=1)
            ):
                raise RuntimeError(
                    "router-neighborhood refinement changed expert widths"
                )
            if any(count <= 0 for count in pair_totals.values()):
                raise RuntimeError(
                    "every router-neighborhood domain must contain pair evidence"
                )
            propagation_mask = base_mask
            layer_coefficients = None
            router_neighborhood_summary = {
                "pair_counts": pair_totals,
                "fallback_experts": fallback_totals,
                "changed_experts": sum(
                    int(record["swap_count"] > 0) for record in records
                ),
                "propagation": "base_checkpoint",
            }
        elif trajectory_field_domains:
            if (
                pareto_base_masks is None
                or general_accumulator is None
                or reasoning_accumulator is None
                or hard_math_accumulator is None
                or code_accumulator is None
                or continuation_accumulator is None
                or termination_accumulator is None
                or continuation_token_mask is None
                or termination_token_mask is None
            ):
                raise RuntimeError(
                    "capability trajectory field inputs are missing"
                )
            base_mask = pareto_base_masks[layer_index].to(device)
            if base_mask.shape != (
                accumulator.expert_count,
                accumulator.channel_count,
            ):
                raise RuntimeError(
                    "capability trajectory base mask has incompatible shape"
                )
            expected_layer_kept = round(
                base_mask.numel() * (1 - layer_prune_ratio)
            )
            if int(base_mask.sum().item()) != expected_layer_kept:
                raise RuntimeError(
                    "capability trajectory base mask violates layer budget"
                )
            tail_start = 2 * len(model.model.layers) // 3
            field_enabled = trajectory_field_layer_enabled(
                layer_index, len(model.model.layers)
            )
            base_indices = tuple(
                torch.nonzero(mask, as_tuple=False).flatten()
                for mask in base_mask
            )
            proposals: list[torch.Tensor] = []
            transition_grams: list[torch.Tensor] = []
            if not field_enabled:
                proposals.extend(index.clone() for index in base_indices)
                trajectory_field_trace = {
                    "proposed_swaps": 0,
                    "accepted": False,
                    "domain_deltas": [],
                    "domain_tolerances": [],
                    "transition_delta": 0.0,
                    "transition_tolerance": 0.0,
                    "status": "prefix_frozen",
                }
                selected_indices = tuple(proposals)
            else:
                for expert_index, initial in enumerate(base_indices):
                    anchor_gram = accumulator.interaction_gram(expert_index)
                    continuation_mean = (
                        continuation_accumulator.routed_activation_mean(
                            expert_index
                        )
                        if int(
                            continuation_accumulator.routed_token_counts[
                                expert_index
                            ].item()
                        )
                        > 0
                        else None
                    )
                    termination_mean = (
                        termination_accumulator.routed_activation_mean(
                            expert_index
                        )
                        if int(
                            termination_accumulator.routed_token_counts[
                                expert_index
                            ].item()
                        )
                        > 0
                        else None
                    )
                    components = build_trajectory_transition_components(
                        anchor_gram,
                        continuation_accumulator.metric_down_projection(
                            expert_index
                        ),
                        problem_mean=None,
                        solution_mean=continuation_mean,
                        termination_mean=termination_mean,
                    )
                    transition = components[2]
                    if transition is None:
                        transition = anchor_gram.float()
                        proposal = initial
                    elif initial.numel() in {
                        0,
                        accumulator.channel_count,
                    }:
                        proposal = initial
                    else:
                        proposal = swap_refine_keep_indices(
                            transition, initial, max_swaps=1
                        )
                    transition_grams.append(transition)
                    proposals.append(proposal)
                if trajectory_field_guarded:
                    domain_accumulators = (
                        general_accumulator,
                        reasoning_accumulator,
                        hard_math_accumulator,
                        code_accumulator,
                    )
                    domain_grams = tuple(
                        tuple(
                            domain.interaction_gram(expert_index)
                            for expert_index in range(accumulator.expert_count)
                        )
                        for domain in domain_accumulators
                    )
                    selected_indices, trajectory_field_trace = (
                        guard_layer_swap_field(
                            domain_grams,
                            tuple(transition_grams),
                            base_indices,
                            tuple(proposals),
                            return_trace=True,
                        )
                    )
                    trajectory_field_trace["status"] = (
                        "field_accepted"
                        if trajectory_field_trace["accepted"]
                        else "field_rejected"
                    )
                else:
                    proposed_swaps = sum(
                        int(
                            len(
                                set(initial.cpu().tolist())
                                - set(proposal.cpu().tolist())
                            )
                        )
                        for initial, proposal in zip(base_indices, proposals)
                    )
                    selected_indices = tuple(proposals)
                    trajectory_field_trace = {
                        "proposed_swaps": proposed_swaps,
                        "accepted": proposed_swaps > 0,
                        "domain_deltas": [],
                        "domain_tolerances": [],
                        "transition_delta": None,
                        "transition_tolerance": None,
                        "status": (
                            "field_accepted_unconstrained"
                            if proposed_swaps > 0
                            else "field_noop_unconstrained"
                        ),
                    }
            keep_mask = torch.zeros_like(base_mask)
            records = []
            for expert_index, (initial, selected) in enumerate(
                zip(base_indices, selected_indices)
            ):
                keep_mask[expert_index, selected] = True
                swap_count = int(
                    (keep_mask[expert_index] != base_mask[expert_index])
                    .sum()
                    .item()
                    // 2
                )
                transition_delta: float | None = None
                transition_tolerance: float | None = None
                if field_enabled:
                    transition_gram = transition_grams[expert_index].float()
                    base_error = gram_reconstruction_error(
                        transition_gram, initial
                    )
                    selected_error = gram_reconstruction_error(
                        transition_gram, selected
                    )
                    transition_delta = float(
                        (selected_error - base_error).item()
                    )
                    transition_tolerance = float(
                        (
                            torch.finfo(torch.float32).eps
                            * base_error.abs().clamp_min(1)
                        ).item()
                    )
                    if (
                        not trajectory_field_guarded
                        and swap_count > 0
                        and transition_delta >= -transition_tolerance
                    ):
                        raise RuntimeError(
                            "unconstrained trajectory swap did not strictly "
                            "improve D-to-T reconstruction"
                        )
                records.append(
                    {
                        "expert": expert_index,
                        "retained_channels": int(initial.numel()),
                        "selection_status": trajectory_field_trace["status"],
                        "swap_budget": (
                            1
                            if layer_index >= tail_start
                            and initial.numel()
                            not in {0, accumulator.channel_count}
                            else 0
                        ),
                        "swap_count": swap_count,
                        "transition_error_delta": transition_delta,
                        "transition_error_tolerance": transition_tolerance,
                        "continuation_routed_tokens": int(
                            continuation_accumulator.routed_token_counts[
                                expert_index
                            ].item()
                        ),
                        "termination_routed_tokens": int(
                            termination_accumulator.routed_token_counts[
                                expert_index
                            ].item()
                        ),
                    }
                )
            if not torch.equal(keep_mask.sum(dim=1), base_mask.sum(dim=1)):
                raise RuntimeError(
                    "capability trajectory field changed expert widths"
                )
            if layer_index < tail_start and not torch.equal(
                keep_mask, base_mask
            ):
                raise RuntimeError(
                    "capability trajectory field changed the frozen prefix"
                )
            propagation_mask = base_mask
            layer_coefficients = None
            trajectory_domain_summary = {
                "assistant_continuation": {
                    "statistics_tokens": int(
                        continuation_token_mask.sum().item()
                    ),
                    "routed_tokens": int(
                        continuation_accumulator.routed_token_counts.sum().item()
                    ),
                },
                "assistant_end": {
                    "statistics_tokens": int(
                        termination_token_mask.sum().item()
                    ),
                    "routed_tokens": int(
                        termination_accumulator.routed_token_counts.sum().item()
                    ),
                },
                "field_guard": trajectory_field_trace,
            }
        elif trajectory_domains:
            if (
                pareto_base_masks is None
                or general_accumulator is None
                or problem_accumulator is None
                or solution_accumulator is None
                or end_accumulator is None
                or response_accumulator is None
                or trajectory_problem_token_mask is None
                or phase_continuation_token_mask is None
                or phase_end_token_mask is None
            ):
                raise RuntimeError("trajectory refinement inputs are missing")
            base_mask = pareto_base_masks[layer_index].to(device)
            keep_mask = base_mask.clone()
            expected_layer_kept = round(
                keep_mask.numel() * (1 - layer_prune_ratio)
            )
            if int(keep_mask.sum().item()) != expected_layer_kept:
                raise RuntimeError("trajectory base mask violates layer budget")
            robust_importance = accumulator.camera_importance(args.camera_alpha)
            records = []
            for expert_index in range(accumulator.expert_count):
                initial = torch.nonzero(
                    base_mask[expert_index], as_tuple=False
                ).flatten()
                anchor_gram = response_accumulator.interaction_gram(expert_index)

                def observed_mean(
                    domain: Qwen3LayerStatsAccumulator,
                ) -> torch.Tensor | None:
                    if int(domain.routed_token_counts[expert_index].item()) == 0:
                        return None
                    return domain.routed_activation_mean(expert_index)

                phase_means = (
                    observed_mean(problem_accumulator),
                    observed_mean(solution_accumulator),
                    observed_mean(end_accumulator),
                )
                transition_components = build_trajectory_transition_components(
                    anchor_gram,
                    solution_accumulator.metric_down_projection(expert_index),
                    problem_mean=phase_means[0],
                    solution_mean=phase_means[1],
                    termination_mean=phase_means[2],
                )
                observed_components = tuple(
                    component
                    for component in transition_components
                    if component is not None
                )
                trajectory_gram = torch.stack(observed_components).mean(dim=0)
                objective = reweight_gram_with_importance(
                    trajectory_gram,
                    robust_importance[expert_index],
                    interaction_strength=args.interaction_strength,
                )
                guard_trace: dict[str, object] | None = None
                if initial.numel() in {0, accumulator.channel_count}:
                    refined = initial
                elif (
                    args.calibration_aggregation
                    == "trajectory_transition_guarded_refinement"
                ):
                    constraint_objectives = tuple(
                        reweight_gram_with_importance(
                            component,
                            robust_importance[expert_index],
                            interaction_strength=args.interaction_strength,
                        )
                        for component in observed_components
                    )
                    refined, guard_trace = guarded_swap_refine_keep_indices(
                        objective,
                        constraint_objectives,
                        initial,
                        return_trace=True,
                    )
                else:
                    refined = swap_refine_keep_indices(
                        objective, initial, max_swaps=1
                    )
                keep_mask[expert_index] = False
                keep_mask[expert_index, refined] = True
                records.append(
                    {
                        "expert": expert_index,
                        "retained_channels": int(initial.numel()),
                        "selection_status": (
                            "trajectory_refined"
                            if not torch.equal(
                                torch.sort(initial).values, refined
                            )
                            else "trajectory_fixed_point"
                        ),
                        "swap_budget": 1 if initial.numel() not in {0, accumulator.channel_count} else 0,
                        "swap_count": int(
                            (base_mask[expert_index] != keep_mask[expert_index])
                            .sum()
                            .item()
                            // 2
                        ),
                        "initial_objective_error": float(
                            gram_reconstruction_error(objective, initial).item()
                        ),
                        "refined_objective_error": float(
                            gram_reconstruction_error(objective, refined).item()
                        ),
                        "guard_trace": guard_trace,
                        "problem_routed_tokens": int(
                            problem_accumulator.routed_token_counts[expert_index].item()
                        ),
                        "solution_routed_tokens": int(
                            solution_accumulator.routed_token_counts[expert_index].item()
                        ),
                        "termination_routed_tokens": int(
                            end_accumulator.routed_token_counts[expert_index].item()
                        ),
                    }
                )
            if not torch.equal(keep_mask.sum(dim=1), base_mask.sum(dim=1)):
                raise RuntimeError("trajectory refinement changed expert widths")
            propagation_mask = base_mask
            layer_coefficients = None
            trajectory_specs = (
                ("problem", problem_accumulator, trajectory_problem_token_mask),
                ("solution_continuation", solution_accumulator, phase_continuation_token_mask),
                ("assistant_end", end_accumulator, phase_end_token_mask),
            )
            trajectory_domain_summary = {}
            for name, domain, selection_mask in trajectory_specs:
                routed = domain.routed_token_counts
                statistics_tokens = int(selection_mask.sum().item())
                if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                    raise RuntimeError(f"trajectory domain {name} has no observations")
                trajectory_domain_summary[name] = {
                    "statistics_tokens": statistics_tokens,
                    "routed_tokens": int(routed.sum().item()),
                    "min_expert_routed_tokens": int(routed.min().item()),
                    "max_expert_routed_tokens": int(routed.max().item()),
                }
        elif pareto_domains:
            if (
                pareto_base_masks is None
                or pareto_hard_full_accumulator is None
                or pareto_hard_solution_accumulator is None
                or general_accumulator is None
                or reasoning_accumulator is None
            ):
                raise RuntimeError("pareto refinement inputs are missing")
            keep_mask = pareto_base_masks[layer_index].to(device).clone()
            if keep_mask.shape != (
                accumulator.expert_count,
                accumulator.channel_count,
            ):
                raise RuntimeError("pareto base mask has incompatible shape")
            expected_layer_kept = round(
                keep_mask.numel() * (1 - layer_prune_ratio)
            )
            if int(keep_mask.sum().item()) != expected_layer_kept:
                raise RuntimeError("pareto base mask violates layer budget")
            records = []
            for expert_index in range(accumulator.expert_count):
                initial = torch.nonzero(
                    keep_mask[expert_index], as_tuple=False
                ).flatten()
                general_gram = (
                    0.9 * general_accumulator.interaction_gram(expert_index)
                    + 0.1
                    * reasoning_accumulator.interaction_gram(expert_index)
                )
                full_gram = pareto_hard_full_accumulator.interaction_gram(
                    expert_index
                )
                solution_gram = (
                    pareto_hard_solution_accumulator.interaction_gram(
                        expert_index
                    )
                )
                grams = (general_gram, full_gram, solution_gram)
                pareto_layer_enabled = (
                    effective_calibration_aggregation
                    == "pareto_residual_refinement"
                )
                swap_budget = (
                    pareto_trust_region_swap_budget(int(initial.numel()))
                    if pareto_layer_enabled
                    else 0
                )
                refined, swap_trace = pareto_swap_refine_keep_indices(
                    grams,
                    initial,
                    max_swaps=swap_budget,
                    return_trace=True,
                )
                keep_mask[expert_index] = False
                keep_mask[expert_index, refined] = True
                initial_errors = [
                    float(gram_reconstruction_error(gram, initial).item())
                    for gram in grams
                ]
                refined_errors = [
                    float(gram_reconstruction_error(gram, refined).item())
                    for gram in grams
                ]
                if any(
                    any(
                        delta > tolerance
                        for delta, tolerance in zip(
                            step["deltas"], step["tolerances"]
                        )
                    )
                    or not any(
                        delta < -tolerance
                        for delta, tolerance in zip(
                            step["deltas"], step["tolerances"]
                        )
                    )
                    for step in swap_trace
                ):
                    raise RuntimeError("pareto trace violated non-increase")
                records.append(
                    {
                        "expert": expert_index,
                        "retained_channels": int(initial.numel()),
                        "selection_status": (
                            "pareto_prefix_frozen"
                            if not pareto_layer_enabled
                            else (
                                "pareto_refined"
                                if swap_trace
                                else "pareto_fixed_point"
                            )
                        ),
                        "swap_budget": swap_budget,
                        "swap_count": len(swap_trace),
                        "initial_objective_errors": initial_errors,
                        "refined_objective_errors": refined_errors,
                        "swap_trace": swap_trace,
                    }
                )
            if not torch.equal(
                keep_mask.sum(dim=1),
                pareto_base_masks[layer_index].sum(dim=1).to(device),
            ):
                raise RuntimeError("pareto refinement changed expert widths")
            layer_coefficients = None
            pareto_specs = (
                ("wikitext_all", general_accumulator, general_sequence_count * args.calib_length),
                (
                    "hardmath_full",
                    pareto_hard_full_accumulator,
                    pareto_hard_math_sequence_count * args.calib_length,
                ),
                (
                    "hardmath_solution",
                    pareto_hard_solution_accumulator,
                    int(
                        statistics_token_mask[
                            general_sequence_count : general_sequence_count
                            + pareto_hard_math_sequence_count
                        ].sum().item()
                    ),
                ),
                (
                    "reasoning_all",
                    reasoning_accumulator,
                    pareto_reasoning_sequence_count * args.calib_length,
                ),
            )
            pareto_domain_summary = {}
            for name, domain_accumulator, statistics_tokens in pareto_specs:
                routed = domain_accumulator.routed_token_counts
                if statistics_tokens <= 0 or int(routed.sum().item()) <= 0:
                    raise RuntimeError(f"pareto domain {name} has no observations")
                pareto_domain_summary[name] = {
                    "statistics_tokens": statistics_tokens,
                    "routed_tokens": int(routed.sum().item()),
                    "min_expert_routed_tokens": int(routed.min().item()),
                    "max_expert_routed_tokens": int(routed.max().item()),
                }
        else:
            keep_mask, records, layer_coefficients = select_layer_mask(
                accumulator,
                method=args.method,
                allocation=args.allocation,
                camera_alpha=args.camera_alpha,
                interaction_strength=args.interaction_strength,
                max_swaps=layer_refinement_budget(
                    args.max_swaps, layer_index, skipped_refine_layers
                ),
                min_channels_per_expert=args.min_channels_per_expert,
                coefficient_ridge_ratio=args.coefficient_ridge_ratio,
                coefficient_clip_min=args.coefficient_clip_min,
                coefficient_clip_max=args.coefficient_clip_max,
                prune_ratio=layer_prune_ratio,
            )
        for record in records:
            record["layer"] = layer_index
            record["layer_refinement_skipped"] = (
                layer_index in skipped_refine_layers
            )
        local_records.extend(records)
        apply_qwen3_keep_mask_(
            moe, keep_mask if propagation_mask is None else propagation_mask
        )
        if (
            layer_coefficients is not None
            and not args.defer_coefficient_application
        ):
            apply_qwen3_channel_coefficients_(moe, layer_coefficients)

        for start in range(0, inputs.shape[0], args.chunk_size):
            stop = start + args.chunk_size
            outputs[start:stop] = decoder_layer_hidden(
                layer,
                inputs[start:stop],
                layer_kwargs,
                layer_index=layer_index,
                batch_start=start,
                replay_pass="propagation",
                captured_batch_size=min(args.chunk_size, inputs.shape[0]),
            )

        per_expert_kept = keep_mask.sum(dim=1)
        layer_summaries.append(
            {
                "layer": layer_index,
                "total_channels": keep_mask.numel(),
                "retained_channels": int(keep_mask.sum().item()),
                "retained_ratio": float(keep_mask.float().mean().item()),
                "prune_ratio": layer_prune_ratio,
                "effective_calibration_aggregation": effective_calibration_aggregation,
                "effective_statistics_region": (
                    (
                        "aligned_dense_loop_onset_predictors"
                        if aligned_loop_state_domains
                        else "train_loop_gold_orthogonal_pairs"
                    )
                    if loop_refinement_domains
                    else (
                        "four_domain_router_neighborhoods"
                        if router_neighborhood_domains
                        else (
                            "capability_response_transition_masks"
                            if trajectory_field_domains
                            else (
                                "trajectory_phase_masks"
                                if trajectory_domains
                                else (
                                    "phase_domain_masks"
                                    if phase_domains
                                    else (
                                        "all_tokens_plus_assistant_end_capacity"
                                        if universal_termination_domains
                                        else (
                                            "all_tokens"
                                            if bool(
                                                effective_statistics_token_mask.all().item()
                                            )
                                            else "calibration_token_mask"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "refinement_propagation": (
                    "base_checkpoint"
                    if (
                        trajectory_domains
                        or trajectory_field_domains
                        or router_neighborhood_domains
                        or loop_refinement_domains
                    )
                    else (
                        "candidate_mask"
                    )
                ),
                "phase_domains": phase_domain_summary,
                "specialist_domains": specialist_domain_summary,
                "pareto_domains": pareto_domain_summary,
                "trajectory_domains": trajectory_domain_summary,
                "router_neighborhood": router_neighborhood_summary,
                "loop_orthogonal": loop_orthogonal_summary,
                "min_retained_per_expert": int(per_expert_kept.min().item()),
                "max_retained_per_expert": int(per_expert_kept.max().item()),
                "mean_retained_per_expert": float(per_expert_kept.float().mean().item()),
                "min_routed_tokens": int(accumulator.routed_token_counts.min().item()),
                "max_routed_tokens": int(accumulator.routed_token_counts.max().item()),
                "elapsed_seconds": time.monotonic() - layer_start,
            }
        )
        masks.append(keep_mask.cpu())
        if layer_coefficients is not None:
            down_coefficients.append(layer_coefficients.cpu())
        layer.cpu()
        del (
            accumulator,
            general_accumulator,
            reasoning_accumulator,
            math_accumulator,
            code_accumulator,
            level4_accumulator,
            level5_accumulator,
            end_accumulator,
            problem_accumulator,
            solution_accumulator,
            response_accumulator,
            phase_math_accumulator,
            phase_domain_accumulators,
            hard_math_accumulator,
            termination_accumulator,
            continuation_accumulator,
            pareto_hard_full_accumulator,
            pareto_hard_solution_accumulator,
            loop_phase_means,
            gold_phase_means,
            loop_phase_routed_counts,
            gold_phase_routed_counts,
            loop_phase_selected_counts,
            gold_phase_selected_counts,
        )
        torch.cuda.empty_cache()
        inputs, outputs = outputs, inputs
        print(
            f"layer {layer_index + 1}/{len(model.model.layers)} complete "
            f"in {layer_summaries[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    retained_channels = sum(int(mask.sum().item()) for mask in masks)
    total_channels = sum(mask.numel() for mask in masks)
    exact_retained_channels = Fraction(total_channels) * (
        1 - Fraction(str(args.prune_ratio))
    )
    if (
        exact_retained_channels.denominator != 1
        or retained_channels != exact_retained_channels.numerator
    ):
        raise RuntimeError(
            "layer budget schedule violated the exact model-wide pruning endpoint"
        )
    if trajectory_field_domains and not any(
        bool(
            layer_summary.get("trajectory_domains", {})
            .get("field_guard", {})
            .get("accepted", False)
        )
        for layer_summary in layer_summaries
    ):
        raise RuntimeError(
            "capability trajectory field accepted no tail layer"
        )
    if router_neighborhood_domains and not any(
        int(record.get("swap_count", 0)) > 0 for record in local_records
    ):
        raise RuntimeError(
            "router-neighborhood refinement produced no channel swap"
        )
    if loop_refinement_domains and not any(
        int(record.get("swap_count", 0)) > 0 for record in local_records
    ):
        raise RuntimeError(
            "self-induced loop refinement produced no channel swap"
        )

    checkpoint = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.revision,
        "model_source": str(Path(model_source).resolve()) if args.model_source else model_source,
        "camera_source_revision": CAMERA_REVISION,
        "method": args.method,
        "allocation": args.allocation,
        "prune_ratio": args.prune_ratio,
        "camera_alpha": (
            args.camera_alpha
            if args.method in {"camera", "ramp_camera"}
            else None
        ),
        "interaction_strength": (
            args.interaction_strength if args.method == "ramp_camera" else None
        ),
        "interaction_metric": args.interaction_metric,
        "difference_regularization": (
            args.difference_regularization
            if args.interaction_metric == "centered_difference"
            else None
        ),
        "difference_strength": (
            args.difference_strength
            if args.interaction_metric == "energy_calibrated_difference"
            else None
        ),
        "max_swaps": args.max_swaps if args.method == "ramp_camera" else None,
        "skip_refine_layers": (
            args.skip_refine_layer if args.method == "ramp_camera" else None
        ),
        "coefficient_ridge_ratio": args.coefficient_ridge_ratio,
        "defer_coefficient_application": args.defer_coefficient_application,
        "coefficient_difference_strength": (
            args.coefficient_difference_strength
            if args.coefficient_ridge_ratio is not None
            else None
        ),
        "coefficient_clip": (
            [args.coefficient_clip_min, args.coefficient_clip_max]
            if args.coefficient_ridge_ratio is not None
            else None
        ),
        "min_channels_per_expert": args.min_channels_per_expert,
        "output_metric": args.output_metric,
        "calibration_corpus": args.calibration_corpus,
        "calibration_aggregation": args.calibration_aggregation,
        "pareto_base_checkpoint": (
            str(args.pareto_base_checkpoint.resolve())
            if args.pareto_base_checkpoint is not None
            else None
        ),
        "layer_budget_schedule": args.layer_budget_schedule,
        "sequential": True,
        "masks": masks,
    }
    if args.coefficient_ridge_ratio is not None:
        if len(down_coefficients) != len(masks):
            raise RuntimeError("missing coefficient tensor for a decoder layer")
        checkpoint["down_coefficients"] = down_coefficients
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = args.mask_output.with_suffix(
        args.mask_output.suffix + ".tmp"
    )
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(args.mask_output)

    result = {
        "schema_version": 1,
        "experiment_type": "qwen3_moe_mask_generation",
        "result_label": "formal",
        "smoke": False,
        "method": args.method,
        "allocation": args.allocation,
        "prune_ratio": args.prune_ratio,
        "model": args.model,
        "model_revision": args.revision,
        "model_source": str(Path(model_source).resolve()) if args.model_source else model_source,
        "camera_source_revision": CAMERA_REVISION,
        "camera_alpha": (
            args.camera_alpha
            if args.method in {"camera", "ramp_camera"}
            else None
        ),
        "interaction_strength": (
            args.interaction_strength if args.method == "ramp_camera" else None
        ),
        "interaction_metric": args.interaction_metric,
        "difference_regularization": (
            args.difference_regularization
            if args.interaction_metric == "centered_difference"
            else None
        ),
        "difference_strength": (
            args.difference_strength
            if args.interaction_metric == "energy_calibrated_difference"
            else None
        ),
        "max_swaps": args.max_swaps if args.method == "ramp_camera" else None,
        "skip_refine_layers": (
            args.skip_refine_layer if args.method == "ramp_camera" else None
        ),
        "coefficient_ridge_ratio": args.coefficient_ridge_ratio,
        "defer_coefficient_application": args.defer_coefficient_application,
        "coefficient_difference_strength": (
            args.coefficient_difference_strength
            if args.coefficient_ridge_ratio is not None
            else None
        ),
        "coefficient_clip": (
            [args.coefficient_clip_min, args.coefficient_clip_max]
            if args.coefficient_ridge_ratio is not None
            else None
        ),
        "min_channels_per_expert": args.min_channels_per_expert,
        "output_metric": args.output_metric,
        "calibration_corpus": args.calibration_corpus,
        "calibration_aggregation": args.calibration_aggregation,
        "pareto_base_checkpoint": (
            str(args.pareto_base_checkpoint.resolve())
            if args.pareto_base_checkpoint is not None
            else None
        ),
        "layer_budget_schedule": args.layer_budget_schedule,
        "tokens_per_expert": args.tokens_per_expert,
        "sequential": True,
        "calibration": calibration,
        "layers": layer_summaries,
        "local_reconstruction": local_records,
        "mask_checkpoint": str(args.mask_output.resolve()),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "cuda_visible_devices": __import__("os").environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
    }
    _json_atomic_write(args.result_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
