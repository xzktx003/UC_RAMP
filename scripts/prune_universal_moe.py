#!/usr/bin/env python3
"""Transfer the frozen universal capability envelope across sparse MoE models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from ramp_moe.moe_adapter import (
    exact_layer_retained_budgets,
    normalization_output_scale,
    resolve_decoder_layers,
    resolve_final_norm,
    resolve_moe_layer,
)
from ramp_moe.qwen3_stats import (
    Qwen3LayerStatsAccumulator,
    UniversalEqualDomainStatsAccumulator,
    UniversalCapabilityEnvelopeStatsAccumulator,
    UniversalTerminationCapacityEnvelopeStatsAccumulator,
)
from ramp_moe.rate_distortion import (
    select_layer_expert_rate_distortion_masks,
    select_layer_expert_rate_distortion_masks_with_layer_budgets,
)
from ramp_moe.selection import apply_qwen3_keep_mask_, select_camera_keep_mask
from scripts.prune_qwen3 import (
    assistant_chat_end_predictor_token_mask,
    calibration_tokens,
    capture_first_layer_inputs,
    decoder_layer_hidden,
    select_ramp_camera_mask_with_expert_widths,
    select_layer_mask,
    specialist_domain_batch_counts,
)


FORMAL_NSAMPLES = 448
CAMERA_FORMAL_NSAMPLES = 128
FORMAL_SEQUENCE_LENGTH = 2048
FORMAL_RESERVOIR_SIZE = 512
SEED = 1234
UNIVERSAL_PROFILES = frozenset({"universal", "universal_termination"})
REVIEW_SINGLE_DOMAIN_PROFILES = frozenset({"review_camera_single_448"})
REVIEW_FOUR_DOMAIN_PROFILES = frozenset(
    {
        "review_camera_four_domain_pooled",
        "review_camera_four_domain_pooled_ramp",
        "review_camera_four_domain_normalized",
        "review_anchor_only",
        "review_stage1_envelope",
        "review_stage1_ramp",
        "review_stage1_ramp_uniform",
    }
)
REVIEW_SPLIT_DOMAIN_PROFILES = REVIEW_FOUR_DOMAIN_PROFILES - frozenset(
    {"review_camera_four_domain_pooled", "review_camera_four_domain_pooled_ramp"}
)
REVIEW_PROFILES = REVIEW_SINGLE_DOMAIN_PROFILES | REVIEW_FOUR_DOMAIN_PROFILES
RATE_DISTORTION_PROFILES = frozenset(
    {
        "rate_distortion",
        "rate_distortion_no_ramp",
        "rate_distortion_anchor_only",
        "rate_distortion_uniform_layer",
        "rate_distortion_raw_energy",
        "rate_distortion_fixed_expert_width",
    }
)
FOUR_DOMAIN_ARTIFACT_PROFILES = (
    UNIVERSAL_PROFILES | RATE_DISTORTION_PROFILES | REVIEW_FOUR_DOMAIN_PROFILES
)
SPLIT_DOMAIN_STATS_PROFILES = (
    UNIVERSAL_PROFILES | RATE_DISTORTION_PROFILES | REVIEW_SPLIT_DOMAIN_PROFILES
)


@dataclass(frozen=True)
class TransferProfileSpec:
    calibration_corpus: str
    calibration_aggregation: str
    method: str
    layer_budget_schedule: str
    layer_budgets: tuple[int, ...]
    nsamples: int
    sequence_length: int
    reservoir_size: int
    interaction_strength: float | None
    max_swaps: int
    domain_names: tuple[str, ...]
    rate_distortion_capacity_signal: str | None
    rate_distortion_budget_scope: str | None
    rate_distortion_score_mode: str | None
    rate_distortion_max_width_delta: int | None


def _exact_retained_budgets(
    schedule: str,
    *,
    layer_count: int,
    expert_count: int,
    channel_count: int,
    pruning_rate: float,
) -> tuple[int, ...]:
    """Allocate deterministic exact endpoint budgets without tuning by model."""

    rate = Fraction(str(pruning_rate))
    if rate not in {Fraction(1, 4), Fraction(1, 2)}:
        raise ValueError("pruning rate must be one of the frozen 0.25/0.50 endpoints")
    if rate == Fraction(1, 2):
        return exact_layer_retained_budgets(
            schedule,
            layer_count=layer_count,
            expert_count=expert_count,
            channel_count=channel_count,
        )
    if layer_count <= 0 or expert_count <= 0 or channel_count <= 0:
        raise ValueError("layer, expert, and channel counts must be positive")
    layer_capacity = expert_count * channel_count
    exact_target = Fraction(layer_count * layer_capacity) * (1 - rate)
    if exact_target.denominator != 1:
        raise ValueError("global capacity cannot represent the pruning endpoint exactly")
    target = exact_target.numerator
    if schedule == "uniform":
        weights = (Fraction(1),) * layer_count
    elif schedule == "tail_protected":
        tail_start = 2 * layer_count // 3
        weights = tuple(
            Fraction(7, 16) if index < tail_start else Fraction(10, 16)
            for index in range(layer_count)
        )
    else:
        raise ValueError(f"unsupported layer budget schedule: {schedule}")
    weight_total = sum(weights, Fraction())
    exact = tuple(Fraction(target) * weight / weight_total for weight in weights)
    budgets = [value.numerator // value.denominator for value in exact]
    remaining = target - sum(budgets)
    remainder_order = sorted(
        range(layer_count),
        key=lambda index: (-(exact[index] - budgets[index]), index),
    )
    for index in remainder_order[:remaining]:
        budgets[index] += 1
    if sum(budgets) != target:
        raise RuntimeError("failed to allocate the exact retained endpoint budget")
    if any(budget < 0 or budget > layer_capacity for budget in budgets):
        raise RuntimeError("normalized layer budget lies outside layer capacity")
    return tuple(budgets)


def _retained_endpoint_label(pruning_rate: float) -> int:
    rate = Fraction(str(pruning_rate))
    return int((1 - rate) * 100)


def _profile_spec(
    profile: str,
    *,
    layer_count: int,
    expert_count: int,
    channel_count: int,
    pruning_rate: float = 0.50,
    smoke: bool,
    smoke_nsamples: int = 49,
    smoke_sequence_length: int = 128,
    smoke_reservoir_size: int = 16,
) -> TransferProfileSpec:
    """Resolve one frozen transfer protocol without architecture tuning."""

    retained_endpoint = _retained_endpoint_label(pruning_rate)
    if profile in REVIEW_PROFILES:
        rate = Fraction(str(pruning_rate))
        supported_exact75 = profile in {
            "review_anchor_only",
            "review_stage1_envelope",
            "review_stage1_ramp",
        } and rate in {
            Fraction(1, 4),
            Fraction(1, 2),
        }
        if rate != Fraction(1, 2) and not supported_exact75:
            raise ValueError(
                "review_v1 fair ablations are frozen at exact50; "
                "review_anchor_only, review_stage1_envelope and "
                "fixed-width review_stage1_ramp "
                "additionally support exact75"
            )
        budget_schedule = (
            "uniform" if profile == "review_stage1_ramp_uniform" else "tail_protected"
        )
        budgets = _exact_retained_budgets(
            budget_schedule,
            layer_count=layer_count,
            expert_count=expert_count,
            channel_count=channel_count,
            pruning_rate=pruning_rate,
        )
        aggregation_by_profile = {
            "review_camera_single_448": "single_domain",
            "review_camera_four_domain_pooled": "four_domain_raw_pooled",
            "review_camera_four_domain_pooled_ramp": "four_domain_raw_pooled",
            "review_camera_four_domain_normalized": (
                "four_domain_mean_normalized_equal_weight"
            ),
            "review_anchor_only": "universal_anchor_only",
            "review_stage1_envelope": "universal_capability_envelope",
            "review_stage1_ramp": "universal_capability_envelope",
            "review_stage1_ramp_uniform": "universal_capability_envelope",
        }
        ramp_enabled = profile in {
            "review_stage1_envelope",
            "review_stage1_ramp",
            "review_stage1_ramp_uniform",
            "review_camera_four_domain_pooled_ramp",
        }
        return TransferProfileSpec(
            calibration_corpus=(
                "wikitext"
                if profile in REVIEW_SINGLE_DOMAIN_PROFILES
                else "reasoning_hardmath_code_append"
            ),
            calibration_aggregation=aggregation_by_profile[profile],
            method="ramp_camera" if ramp_enabled else "camera",
            layer_budget_schedule=f"{budget_schedule}_normalized_exact{retained_endpoint}",
            layer_budgets=tuple(budgets),
            nsamples=smoke_nsamples if smoke else FORMAL_NSAMPLES,
            sequence_length=(
                smoke_sequence_length if smoke else FORMAL_SEQUENCE_LENGTH
            ),
            reservoir_size=(
                smoke_reservoir_size if smoke else FORMAL_RESERVOIR_SIZE
            ),
            interaction_strength=0.2 if ramp_enabled else None,
            max_swaps=(
                1
                if profile
                in {
                    "review_stage1_ramp",
                    "review_stage1_ramp_uniform",
                    "review_camera_four_domain_pooled_ramp",
                }
                else 0
            ),
            domain_names=(
                ("wikitext",)
                if profile in REVIEW_SINGLE_DOMAIN_PROFILES
                else (
                    ("pooled_four_domain",)
                    if profile
                    in {
                        "review_camera_four_domain_pooled",
                        "review_camera_four_domain_pooled_ramp",
                    }
                    else ("wikitext", "reasoning", "hard_math", "code")
                )
            ),
            rate_distortion_capacity_signal=None,
            rate_distortion_budget_scope=None,
            rate_distortion_score_mode=None,
            rate_distortion_max_width_delta=None,
        )
    if profile in UNIVERSAL_PROFILES or profile in RATE_DISTORTION_PROFILES:
        budget_schedule = (
            "tail_protected"
            if profile in UNIVERSAL_PROFILES
            else "uniform"
        )
        budgets = _exact_retained_budgets(
            budget_schedule,
            layer_count=layer_count,
            expert_count=expert_count,
            channel_count=channel_count,
            pruning_rate=pruning_rate,
        )
        return TransferProfileSpec(
            calibration_corpus="reasoning_hardmath_code_append",
            calibration_aggregation=(
                "universal_termination_capacity_envelope"
                if profile == "universal_termination"
                else "universal_capability_envelope"
            ),
            method="ramp_camera",
            layer_budget_schedule=(
                f"tail_protected_normalized_exact{retained_endpoint}"
                if profile in UNIVERSAL_PROFILES
                else f"layer_expert_rate_distortion_exact{retained_endpoint}"
            ),
            layer_budgets=tuple(budgets),
            nsamples=smoke_nsamples if smoke else FORMAL_NSAMPLES,
            sequence_length=(
                smoke_sequence_length if smoke else FORMAL_SEQUENCE_LENGTH
            ),
            reservoir_size=(
                smoke_reservoir_size if smoke else FORMAL_RESERVOIR_SIZE
            ),
            interaction_strength=0.2,
            max_swaps=0 if profile == "rate_distortion_no_ramp" else 1,
            domain_names=(
                "wikitext",
                "reasoning",
                "hard_math",
                "code",
                *(("assistant_end",) if profile == "universal_termination" else ()),
            ),
            rate_distortion_capacity_signal=(
                None
                if profile in UNIVERSAL_PROFILES
                else (
                    "anchor"
                    if profile == "rate_distortion_anchor_only"
                    else "capability_envelope"
                )
            ),
            rate_distortion_budget_scope=(
                None
                if profile in UNIVERSAL_PROFILES
                else (
                    "layer"
                    if profile == "rate_distortion_uniform_layer"
                    else "model"
                )
            ),
            rate_distortion_score_mode=(
                None
                if profile in UNIVERSAL_PROFILES
                else (
                    "raw" if profile == "rate_distortion_raw_energy" else "lamp"
                )
            ),
            rate_distortion_max_width_delta=(
                None
                if profile in UNIVERSAL_PROFILES
                else (
                    0
                    if profile == "rate_distortion_fixed_expert_width"
                    else max(1, channel_count // 8)
                )
            ),
        )
    if profile == "camera":
        budgets = _exact_retained_budgets(
            "uniform",
            layer_count=layer_count,
            expert_count=expert_count,
            channel_count=channel_count,
            pruning_rate=pruning_rate,
        )
        return TransferProfileSpec(
            calibration_corpus="wikitext",
            calibration_aggregation="pooled",
            method="camera",
            layer_budget_schedule=f"uniform_exact{retained_endpoint}",
            layer_budgets=budgets,
            nsamples=smoke_nsamples if smoke else CAMERA_FORMAL_NSAMPLES,
            sequence_length=(
                smoke_sequence_length if smoke else FORMAL_SEQUENCE_LENGTH
            ),
            reservoir_size=(
                smoke_reservoir_size if smoke else FORMAL_RESERVOIR_SIZE
            ),
            interaction_strength=None,
            max_swaps=0,
            domain_names=("wikitext",),
            rate_distortion_capacity_signal=None,
            rate_distortion_budget_scope=None,
            rate_distortion_score_mode=None,
            rate_distortion_max_width_delta=None,
        )
    raise ValueError(f"unsupported transfer profile: {profile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=(
            "universal",
            "universal_termination",
            "rate_distortion",
            "rate_distortion_no_ramp",
            "rate_distortion_anchor_only",
            "rate_distortion_uniform_layer",
            "rate_distortion_raw_energy",
            "rate_distortion_fixed_expert_width",
            "camera",
            "review_camera_single_448",
            "review_camera_four_domain_pooled",
            "review_camera_four_domain_pooled_ramp",
            "review_camera_four_domain_normalized",
            "review_anchor_only",
            "review_stage1_envelope",
            "review_stage1_ramp",
            "review_stage1_ramp_uniform",
        ),
        default="universal",
    )
    parser.add_argument(
        "--pruning-rate",
        type=float,
        choices=(0.25, 0.50),
        default=0.50,
        help="frozen fraction of MoE channels removed",
    )
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--competition-math-source", type=Path)
    parser.add_argument(
        "--competition-math-exclusion-manifest", type=Path
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--camera-alpha", type=float, default=0.9)
    parser.add_argument("--anchor-general-weight", type=float, default=0.9)
    parser.add_argument("--trust-radius-divisor", type=int, default=8)
    parser.add_argument("--interaction-strength-override", type=float)
    parser.add_argument("--fixed-stage1-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-layers", type=int, default=1)
    parser.add_argument("--smoke-nsamples", type=int, default=49)
    parser.add_argument("--smoke-sequence-length", type=int, default=128)
    parser.add_argument("--smoke-reservoir-size", type=int, default=16)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not 0 <= args.camera_alpha <= 1:
        parser.error("--camera-alpha must be in [0, 1]")
    if not 0 <= args.anchor_general_weight <= 1:
        parser.error("--anchor-general-weight must be in [0, 1]")
    if args.trust_radius_divisor <= 0:
        parser.error("--trust-radius-divisor must be positive")
    if args.interaction_strength_override is not None and not (
        0 <= args.interaction_strength_override <= 1
    ):
        parser.error("--interaction-strength-override must be in [0, 1]")
    if (
        args.fixed_stage1_checkpoint is not None
        and args.profile != "review_stage1_ramp"
    ):
        parser.error(
            "--fixed-stage1-checkpoint requires --profile review_stage1_ramp"
        )
    if (
        args.profile == "review_stage1_ramp"
        and args.pruning_rate == 0.25
        and args.fixed_stage1_checkpoint is None
    ):
        parser.error(
            "review_stage1_ramp at prune-25 requires "
            "--fixed-stage1-checkpoint"
        )
    hard_math_paths = (
        args.competition_math_source,
        args.competition_math_exclusion_manifest,
    )
    if args.profile in FOUR_DOMAIN_ARTIFACT_PROFILES and any(
        path is None for path in hard_math_paths
    ):
        parser.error(
            "capability profiles require competition MATH source and exclusion manifest"
        )
    if args.profile not in FOUR_DOMAIN_ARTIFACT_PROFILES and any(
        path is not None for path in hard_math_paths
    ):
        parser.error("single-domain camera profiles do not consume competition MATH artifacts")
    if args.smoke:
        if args.smoke_layers <= 0:
            parser.error("--smoke-layers must be positive")
        if args.profile in FOUR_DOMAIN_ARTIFACT_PROFILES and (
            args.smoke_nsamples < 49 or args.smoke_nsamples % 7
        ):
            parser.error(
                "--smoke-nsamples must be at least 49 and divisible by 7"
            )
        if (
            args.profile not in FOUR_DOMAIN_ARTIFACT_PROFILES
            and args.smoke_nsamples <= 0
        ):
            parser.error("camera --smoke-nsamples must be positive")
        if args.smoke_sequence_length <= 0 or args.smoke_reservoir_size <= 0:
            parser.error("smoke sequence length and reservoir size must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixed_stage1_masks(
    path: Path,
    *,
    model_name: str,
    layer_count: int,
    expert_count: int,
    channel_count: int,
    layer_limit: int,
) -> tuple[tuple[torch.Tensor, ...], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("fixed Stage-I checkpoint must contain a mapping")
    if payload.get("model") != model_name:
        raise ValueError("fixed Stage-I checkpoint model mismatch")
    if payload.get("calibration_aggregation") != "universal_capability_envelope":
        raise ValueError("fixed Stage-I checkpoint aggregation mismatch")
    if int(payload.get("max_swaps", -1)) != 0:
        raise ValueError("fixed Stage-I checkpoint must be a no-swap solution")
    masks = payload.get("masks")
    if not isinstance(masks, list) or len(masks) != layer_count:
        raise ValueError("fixed Stage-I checkpoint layer count mismatch")
    normalized = []
    for index, mask in enumerate(masks):
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"fixed Stage-I layer {index} mask must be boolean")
        if mask.shape != (expert_count, channel_count):
            raise ValueError(f"fixed Stage-I layer {index} mask shape mismatch")
        normalized.append(mask.cpu())
    return tuple(normalized[:layer_limit]), _sha256(path)


def _next_output_scale(
    model: torch.nn.Module,
    layers: list[torch.nn.Module] | torch.nn.ModuleList,
    layer_index: int,
) -> torch.Tensor:
    norm = (
        layers[layer_index + 1].input_layernorm
        if layer_index + 1 < len(layers)
        else resolve_final_norm(model)
    )
    return normalization_output_scale(norm)


def _capture_routed_inputs(
    view: object,
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    pending: dict[str, torch.Tensor] = {}

    if not bool(getattr(view, "packed", True)):
        def capture_shared(
            _module: torch.nn.Module, hook_inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if pending:
                raise RuntimeError("legacy MoE input capture was not consumed")
            pending["router"] = hook_inputs[0]
            pending["experts"] = hook_inputs[0]

        return pending, [
            view.mask_module.register_forward_pre_hook(capture_shared)
        ]

    def capture_router(
        _module: torch.nn.Module, hook_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        if "router" in pending:
            raise RuntimeError("router input capture was not consumed")
        pending["router"] = hook_inputs[0]

    def capture_experts(
        _module: torch.nn.Module, hook_inputs: tuple[torch.Tensor, ...]
    ) -> None:
        if "experts" in pending:
            raise RuntimeError("expert input capture was not consumed")
        pending["experts"] = hook_inputs[0]

    handles = [
        view.router.register_forward_pre_hook(capture_router),
        view.experts.register_forward_pre_hook(capture_experts),
    ]
    return pending, handles


def _pop_routed_inputs(
    pending: dict[str, torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    router = pending.pop("router", None)
    experts = pending.pop("experts", None)
    if router is None or experts is None or pending:
        raise RuntimeError("failed to capture exactly one router and expert input")
    expected_tokens = batch_size * sequence_length
    if router.numel() != expected_tokens * hidden_size:
        raise RuntimeError("captured router input has an incompatible shape")
    if experts.numel() != expected_tokens * hidden_size:
        raise RuntimeError("captured expert input has an incompatible shape")
    return (
        router.reshape(batch_size, sequence_length, hidden_size),
        experts.reshape(batch_size, sequence_length, hidden_size),
    )


def _release_replay_buffers_(*buffers: torch.Tensor) -> None:
    """Drop stale replay storage before allocating a fresh first-layer pass."""

    for buffer in buffers:
        buffer.set_(
            torch.empty(0, dtype=buffer.dtype, device=buffer.device)
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _universal_termination_token_mask(
    input_ids: torch.Tensor,
    calibration: dict[str, object],
    *,
    general_count: int,
    reasoning_count: int,
    hard_math_count: int,
) -> torch.Tensor:
    """Build IDEA-058 assistant-end support over the three specialist domains."""

    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.ndim != 2:
        raise ValueError("universal termination input ids must be two-dimensional")
    specialist_start = general_count
    reasoning_stop = specialist_start + reasoning_count
    hard_math_stop = reasoning_stop + hard_math_count
    if not 0 <= specialist_start <= reasoning_stop <= hard_math_stop <= len(
        input_ids
    ):
        raise ValueError("universal termination domain counts exceed calibration")
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
    termination = torch.zeros_like(input_ids, dtype=torch.bool)
    termination[specialist_start:] = assistant_chat_end_predictor_token_mask(
        input_ids[specialist_start:],
        assistant_marker=torch.tensor(assistant_ids, dtype=torch.long),
        end_marker=torch.tensor(end_ids, dtype=torch.long),
    )
    for name, domain_mask in (
        ("reasoning", termination[specialist_start:reasoning_stop]),
        ("hard_math", termination[reasoning_stop:hard_math_stop]),
        ("code", termination[hard_math_stop:]),
    ):
        if not bool(domain_mask.any().item()):
            raise RuntimeError(
                f"universal termination domain {name} has no end targets"
            )
    return termination


def _update_universal_statistics(
    router_inputs: torch.Tensor,
    expert_inputs: torch.Tensor,
    *,
    domain_accumulators: tuple[Qwen3LayerStatsAccumulator, ...],
    domain_counts: tuple[int, ...],
    batch_start: int,
    termination_accumulator: Qwen3LayerStatsAccumulator | None,
    termination_token_mask: torch.Tensor | None,
) -> None:
    """Update four ordinary domains and an optional capacity-only end domain."""

    if len(domain_accumulators) != 4 or len(domain_counts) != 4:
        raise ValueError("universal statistics require four ordinary domains")
    if sum(domain_counts) != router_inputs.shape[0]:
        raise ValueError("universal domain counts must cover the routed batch")
    domain_start = 0
    for domain_accumulator, domain_count in zip(
        domain_accumulators, domain_counts
    ):
        domain_stop = domain_start + domain_count
        if domain_count:
            domain_accumulator.update(
                router_inputs[domain_start:domain_stop],
                expert_hidden_states=expert_inputs[domain_start:domain_stop],
            )
        domain_start = domain_stop
    if termination_accumulator is None:
        if termination_token_mask is not None:
            raise ValueError("termination mask requires a termination accumulator")
        return
    if termination_token_mask is None:
        raise ValueError("termination accumulator requires a termination mask")
    specialist_start = domain_counts[0]
    if specialist_start < router_inputs.shape[0]:
        batch_stop = batch_start + router_inputs.shape[0]
        termination_accumulator.update(
            router_inputs[specialist_start:],
            expert_hidden_states=expert_inputs[specialist_start:],
            token_selection_mask=termination_token_mask[
                batch_start + specialist_start : batch_stop
            ],
        )


@torch.no_grad()
def _collect_layer_statistics(
    layer: torch.nn.Module,
    view: object,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    replay: object,
    *,
    layer_index: int,
    sequence_length: int,
    chunk_size: int,
    accumulator: Qwen3LayerStatsAccumulator
    | UniversalCapabilityEnvelopeStatsAccumulator
    | UniversalTerminationCapacityEnvelopeStatsAccumulator,
    domain_accumulators: tuple[Qwen3LayerStatsAccumulator, ...],
    general_count: int,
    reasoning_count: int,
    hard_math_count: int,
    universal: bool,
    termination_accumulator: Qwen3LayerStatsAccumulator | None = None,
    termination_token_mask: torch.Tensor | None = None,
) -> None:
    """Collect one layer's statistics and dense/current propagation output."""

    for start in range(0, inputs.shape[0], chunk_size):
        stop = min(start + chunk_size, inputs.shape[0])
        pending, handles = _capture_routed_inputs(view)
        try:
            outputs[start:stop] = decoder_layer_hidden(
                layer,
                inputs[start:stop],
                replay,
                layer_index=layer_index,
                batch_start=start,
                replay_pass="statistics",
                captured_batch_size=min(chunk_size, inputs.shape[0]),
            )
        finally:
            for handle in handles:
                handle.remove()
        router_inputs, expert_inputs = _pop_routed_inputs(
            pending,
            batch_size=stop - start,
            sequence_length=sequence_length,
            hidden_size=view.hidden_size,
        )
        if universal:
            counts = specialist_domain_batch_counts(
                start,
                stop,
                general_count,
                reasoning_count,
                hard_math_count,
            )
            _update_universal_statistics(
                router_inputs,
                expert_inputs,
                domain_accumulators=domain_accumulators,
                domain_counts=counts,
                batch_start=start,
                termination_accumulator=termination_accumulator,
                termination_token_mask=termination_token_mask,
            )
        else:
            accumulator.update(
                router_inputs,
                expert_hidden_states=expert_inputs,
            )


def _rate_distortion_signals(
    accumulator: UniversalCapabilityEnvelopeStatsAccumulator,
    *,
    camera_alpha: float,
    capacity_signal: str,
    pruning_rate: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the anchor and capacity signals at a nominal half-width."""

    domain_names = ("wikitext", "reasoning", "hard_math", "code")
    domain_diagnostics = []
    for name, domain in zip(domain_names, accumulator.domains):
        importance = domain.camera_importance(camera_alpha)
        domain_diagnostics.append(
            {
                "name": name,
                "importance_mean": float(importance.mean().item()),
                "routed_tokens": int(domain.routed_token_counts.sum().item()),
            }
        )
    invalid = [
        value
        for value in domain_diagnostics
        if not math.isfinite(value["importance_mean"])
        or value["importance_mean"] <= 0
        or value["routed_tokens"] <= 0
    ]
    if invalid:
        raise RuntimeError(
            "rate-distortion profiling found an empty capability domain: "
            f"{domain_diagnostics}"
        )
    anchor = accumulator.camera_importance(camera_alpha)
    if capacity_signal == "anchor":
        return anchor, anchor
    if capacity_signal != "capability_envelope":
        raise ValueError(f"unsupported rate-distortion signal: {capacity_signal}")
    reference = select_camera_keep_mask(
        anchor,
        prune_ratio=pruning_rate,
        allocation="global",
        min_keep_per_expert=0,
    )
    allocation = accumulator.allocation_importance_for_reference_widths(
        camera_alpha, reference.sum(dim=1)
    )
    return anchor, allocation


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("universal pruning requires one visible CUDA GPU")
    device = torch.device("cuda:0")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    print(f"loading local model {args.model_source} on CPU", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_source,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", model.config)
    text_config.use_cache = False
    layers = resolve_decoder_layers(model)
    views = tuple(resolve_moe_layer(layer) for layer in layers)
    first_view = views[0]
    if any(
        (view.expert_count, view.channel_count, view.hidden_size)
        != (
            first_view.expert_count,
            first_view.channel_count,
            first_view.hidden_size,
        )
        for view in views
    ):
        raise RuntimeError("all sparse decoder layers must share expert dimensions")
    profile = _profile_spec(
        args.profile,
        layer_count=len(layers),
        expert_count=first_view.expert_count,
        channel_count=first_view.channel_count,
        pruning_rate=args.pruning_rate,
        smoke=args.smoke,
        smoke_nsamples=args.smoke_nsamples,
        smoke_sequence_length=args.smoke_sequence_length,
        smoke_reservoir_size=args.smoke_reservoir_size,
    )
    nsamples = profile.nsamples
    sequence_length = profile.sequence_length
    reservoir_size = profile.reservoir_size
    budgets = profile.layer_budgets
    layer_limit = min(args.smoke_layers, len(layers)) if args.smoke else len(layers)

    print(f"building fixed {profile.calibration_corpus} calibration", flush=True)
    input_ids, calibration, statistics_mask = calibration_tokens(
        str(args.model_source),
        None,
        nsamples=nsamples,
        sequence_length=sequence_length,
        seed=SEED,
        calibration_corpus=profile.calibration_corpus,
        local_files_only=True,
        competition_math_source=(
            args.competition_math_source
            if args.profile in FOUR_DOMAIN_ARTIFACT_PROFILES
            else None
        ),
        competition_math_exclusion_manifest=(
            args.competition_math_exclusion_manifest
            if args.profile in FOUR_DOMAIN_ARTIFACT_PROFILES
            else None
        ),
    )
    if not bool(statistics_mask.all().item()):
        raise RuntimeError("universal capability transfer requires all-token statistics")
    allocation = calibration["sample_allocation"]
    general_count = int(allocation["wikitext"])
    reasoning_count = int(allocation.get("gsm8k", 0)) + int(
        allocation.get("math_qa", 0)
    )
    hard_math_count = int(allocation.get("competition_math", 0))
    termination_token_mask: torch.Tensor | None = None
    if profile.calibration_aggregation == (
        "universal_termination_capacity_envelope"
    ):
        termination_token_mask = _universal_termination_token_mask(
            input_ids,
            calibration,
            general_count=general_count,
            reasoning_count=reasoning_count,
            hard_math_count=hard_math_count,
        )
        calibration["assistant_end_statistics_tokens"] = int(
            termination_token_mask.sum().item()
        )

    print("capturing layer-specific replay inputs", flush=True)
    inputs, replay = capture_first_layer_inputs(
        model,
        input_ids,
        device=device,
        chunk_size=args.chunk_size,
    )
    outputs = torch.empty_like(inputs)
    if termination_token_mask is not None:
        termination_token_mask = termination_token_mask.to(device)
    rate_distortion_trace: dict[str, object] | None = None
    preallocated_widths: tuple[torch.Tensor, ...] | None = None
    preallocated_initial_masks: tuple[torch.Tensor, ...] | None = None
    fixed_stage1_checkpoint_sha256: str | None = None
    if args.fixed_stage1_checkpoint is not None:
        preallocated_initial_masks, fixed_stage1_checkpoint_sha256 = (
            _load_fixed_stage1_masks(
                args.fixed_stage1_checkpoint,
                model_name=args.model_name,
                layer_count=len(layers),
                expert_count=first_view.expert_count,
                channel_count=first_view.channel_count,
                layer_limit=layer_limit,
            )
        )
        preallocated_widths = tuple(
            mask.sum(dim=1) for mask in preallocated_initial_masks
        )
        if sum(int(widths.sum().item()) for widths in preallocated_widths) != sum(
            budgets[:layer_limit]
        ):
            raise ValueError("fixed Stage-I widths violate the target layer budgets")
        calibration["fixed_stage1_checkpoint_sha256"] = (
            fixed_stage1_checkpoint_sha256
        )
    if args.profile in RATE_DISTORTION_PROFILES:
        print(
            "profiling dense layer-expert rate-distortion signals",
            flush=True,
        )
        anchors: list[torch.Tensor] = []
        allocation_signals: list[torch.Tensor] = []
        for layer_index in range(layer_limit):
            layer = layers[layer_index]
            layer.to(device)
            view = resolve_moe_layer(layer)
            output_scale = _next_output_scale(model, layers, layer_index)

            def make_profile_accumulator(
                seed_offset: int,
            ) -> Qwen3LayerStatsAccumulator:
                return Qwen3LayerStatsAccumulator(
                    view.mask_module,
                    reservoir_size=reservoir_size,
                    reservoir_seed=SEED + layer_index + seed_offset,
                    output_metric_scale=output_scale,
                    interaction_metric="second_moment",
                )

            profile_domains = (
                make_profile_accumulator(0),
                make_profile_accumulator(100_000),
                make_profile_accumulator(200_000),
                make_profile_accumulator(300_000),
            )
            profile_accumulator = UniversalCapabilityEnvelopeStatsAccumulator(
                *profile_domains,
                general_weight=args.anchor_general_weight,
            )
            _collect_layer_statistics(
                layer,
                view,
                inputs,
                outputs,
                replay,
                layer_index=layer_index,
                sequence_length=sequence_length,
                chunk_size=args.chunk_size,
                accumulator=profile_accumulator,
                domain_accumulators=profile_domains,
                general_count=general_count,
                reasoning_count=reasoning_count,
                hard_math_count=hard_math_count,
                universal=True,
            )
            anchor, allocation_signal = _rate_distortion_signals(
                profile_accumulator,
                camera_alpha=args.camera_alpha,
                capacity_signal=profile.rate_distortion_capacity_signal or "",
                pruning_rate=args.pruning_rate,
            )
            anchors.append(anchor.cpu())
            allocation_signals.append(allocation_signal.cpu())
            layer.cpu()
            del profile_accumulator, profile_domains
            torch.cuda.empty_cache()
            inputs, outputs = outputs, inputs
            print(
                f"rate-distortion profile layer {layer_index + 1}/{layer_limit} complete",
                flush=True,
            )

        profiled_retained = sum(profile.layer_budgets[:layer_limit])
        allocation_kwargs = {
            "reference_importances": anchors,
            "nominal_prune_ratio": args.pruning_rate,
            "max_width_delta": profile.rate_distortion_max_width_delta,
            "score_mode": profile.rate_distortion_score_mode or "",
        }
        if profile.rate_distortion_budget_scope == "model":
            allocation_masks, rate_distortion_trace = (
                select_layer_expert_rate_distortion_masks(
                    allocation_signals,
                    global_keep_count=profiled_retained,
                    **allocation_kwargs,
                )
            )
        elif profile.rate_distortion_budget_scope == "layer":
            allocation_masks, rate_distortion_trace = (
                select_layer_expert_rate_distortion_masks_with_layer_budgets(
                    allocation_signals,
                    layer_keep_counts=profile.layer_budgets[:layer_limit],
                    **allocation_kwargs,
                )
            )
        else:
            raise RuntimeError("rate-distortion profile has no budget scope")
        preallocated_widths = tuple(mask.sum(dim=1) for mask in allocation_masks)
        calibration["rate_distortion_allocation"] = rate_distortion_trace
        print("recapturing first-layer states for sequential pruning", flush=True)
        _release_replay_buffers_(inputs, outputs)
        del replay
        inputs, replay = capture_first_layer_inputs(
            model,
            input_ids,
            device=device,
            chunk_size=args.chunk_size,
        )
        outputs = torch.empty_like(inputs)

    masks: list[torch.Tensor] = []
    layer_summaries: list[dict[str, object]] = []
    local_records: list[dict[str, object]] = []
    for layer_index in range(layer_limit):
        layer_started = time.monotonic()
        layer = layers[layer_index]
        layer.to(device)
        view = resolve_moe_layer(layer)
        output_scale = _next_output_scale(model, layers, layer_index)

        def make_accumulator(seed_offset: int) -> Qwen3LayerStatsAccumulator:
            return Qwen3LayerStatsAccumulator(
                view.mask_module,
                reservoir_size=reservoir_size,
                reservoir_seed=SEED + layer_index + seed_offset,
                output_metric_scale=output_scale,
                interaction_metric="second_moment",
            )

        termination_accumulator: Qwen3LayerStatsAccumulator | None = None
        if args.profile in SPLIT_DOMAIN_STATS_PROFILES:
            domain_accumulators = (
                make_accumulator(0),
                make_accumulator(100_000),
                make_accumulator(200_000),
                make_accumulator(300_000),
            )
            termination_accumulator = (
                make_accumulator(400_000)
                if args.profile == "universal_termination"
                else None
            )
            accumulator: (
                Qwen3LayerStatsAccumulator
                | UniversalEqualDomainStatsAccumulator
                | UniversalCapabilityEnvelopeStatsAccumulator
                | UniversalTerminationCapacityEnvelopeStatsAccumulator
            ) = (
                UniversalTerminationCapacityEnvelopeStatsAccumulator(
                    *domain_accumulators,
                    termination_accumulator,
                )
                if termination_accumulator is not None
                else (
                    UniversalEqualDomainStatsAccumulator(*domain_accumulators)
                    if args.profile == "review_camera_four_domain_normalized"
                    else UniversalCapabilityEnvelopeStatsAccumulator(
                        *domain_accumulators,
                        general_weight=args.anchor_general_weight,
                    )
                )
            )
        else:
            accumulator = make_accumulator(0)
            domain_accumulators = (accumulator,)
        _collect_layer_statistics(
            layer,
            view,
            inputs,
            outputs,
            replay,
            layer_index=layer_index,
            sequence_length=sequence_length,
            chunk_size=args.chunk_size,
            accumulator=accumulator,
            domain_accumulators=domain_accumulators,
            general_count=general_count,
            reasoning_count=reasoning_count,
            hard_math_count=hard_math_count,
            universal=args.profile in SPLIT_DOMAIN_STATS_PROFILES,
            termination_accumulator=termination_accumulator,
            termination_token_mask=termination_token_mask,
        )

        layer_capacity = view.expert_count * view.channel_count
        retained_budget = (
            int(preallocated_widths[layer_index].sum().item())
            if preallocated_widths is not None
            else budgets[layer_index]
        )
        prune_ratio = 1 - retained_budget / layer_capacity
        interaction_strength = (
            profile.interaction_strength
            if args.interaction_strength_override is None
            else args.interaction_strength_override
        )
        if preallocated_widths is not None:
            keep_mask, records, coefficients = (
                select_ramp_camera_mask_with_expert_widths(
                    accumulator,
                    preallocated_widths[layer_index],
                    camera_alpha=args.camera_alpha,
                    interaction_strength=interaction_strength or 0.0,
                    max_swaps=profile.max_swaps,
                    coefficient_ridge_ratio=None,
                    coefficient_clip_min=0.0,
                    coefficient_clip_max=2.0,
                    initial_keep_mask=(
                        None
                        if preallocated_initial_masks is None
                        else preallocated_initial_masks[layer_index]
                    ),
                )
            )
        else:
            keep_mask, records, coefficients = select_layer_mask(
                accumulator,
                method=profile.method,
                allocation="global",
                camera_alpha=args.camera_alpha,
                interaction_strength=interaction_strength or 0.0,
                max_swaps=profile.max_swaps,
                min_channels_per_expert=0,
                coefficient_ridge_ratio=None,
                coefficient_clip_min=0.0,
                coefficient_clip_max=2.0,
                prune_ratio=prune_ratio,
                max_width_delta_divisor=args.trust_radius_divisor,
            )
        if coefficients is not None:
            raise RuntimeError("frozen universal method must not recover coefficients")
        if int(keep_mask.sum().item()) != retained_budget:
            raise RuntimeError("layer selector violated its exact retained budget")
        for record in records:
            record["layer"] = layer_index
        local_records.extend(records)
        apply_qwen3_keep_mask_(view.mask_module, keep_mask)

        for start in range(0, inputs.shape[0], args.chunk_size):
            stop = min(start + args.chunk_size, inputs.shape[0])
            outputs[start:stop] = decoder_layer_hidden(
                layer,
                inputs[start:stop],
                replay,
                layer_index=layer_index,
                batch_start=start,
                replay_pass="propagation",
                captured_batch_size=min(args.chunk_size, inputs.shape[0]),
            )
        summary_accumulators = (
            (*domain_accumulators, termination_accumulator)
            if termination_accumulator is not None
            else domain_accumulators
        )
        domain_summary = {}
        for name, domain in zip(profile.domain_names, summary_accumulators):
            routed = domain.routed_token_counts
            if int(routed.sum().item()) <= 0:
                raise RuntimeError(f"domain {name} has no routed observations")
            domain_summary[name] = {
                "routed_tokens": int(routed.sum().item()),
                "min_expert_routed_tokens": int(routed.min().item()),
                "max_expert_routed_tokens": int(routed.max().item()),
            }
        per_expert = keep_mask.sum(dim=1)
        layer_summaries.append(
            {
                "layer": layer_index,
                "retained_channels": retained_budget,
                "available_channels": layer_capacity,
                "retained_ratio": retained_budget / layer_capacity,
                "prune_ratio": prune_ratio,
                "min_retained_per_expert": int(per_expert.min().item()),
                "max_retained_per_expert": int(per_expert.max().item()),
                "mean_retained_per_expert": float(per_expert.float().mean().item()),
                "domains": domain_summary,
                "elapsed_seconds": time.monotonic() - layer_started,
            }
        )
        masks.append(keep_mask.cpu())
        layer.cpu()
        del accumulator, domain_accumulators, termination_accumulator
        torch.cuda.empty_cache()
        inputs, outputs = outputs, inputs
        print(
            f"layer {layer_index + 1}/{layer_limit} complete in "
            f"{layer_summaries[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    retained = sum(int(mask.sum().item()) for mask in masks)
    available = sum(mask.numel() for mask in masks)
    status = "smoke_complete" if args.smoke else "complete"
    exact_retained = Fraction(available) * (
        1 - Fraction(str(args.pruning_rate))
    )
    if not args.smoke and (
        exact_retained.denominator != 1 or retained != exact_retained.numerator
    ):
        raise RuntimeError("formal universal mask violated its exact endpoint")
    checkpoint_payload = {
        "schema_version": 1,
        "model": args.model_name,
        "model_revision": args.model_revision,
        "method": profile.method,
        "allocation": "global",
        "prune_ratio": args.pruning_rate,
        "camera_alpha": args.camera_alpha,
        "anchor_general_weight": args.anchor_general_weight,
        "trust_radius_divisor": args.trust_radius_divisor,
        "interaction_strength": (
            profile.interaction_strength
            if args.interaction_strength_override is None
            else args.interaction_strength_override
        ),
        "fixed_stage1_checkpoint": (
            None
            if args.fixed_stage1_checkpoint is None
            else str(args.fixed_stage1_checkpoint.resolve())
        ),
        "fixed_stage1_checkpoint_sha256": fixed_stage1_checkpoint_sha256,
        "max_swaps": profile.max_swaps,
        "calibration_aggregation": profile.calibration_aggregation,
        "layer_budget_schedule": profile.layer_budget_schedule,
        "layer_budgets": list(profile.layer_budgets),
        "rate_distortion_allocation": rate_distortion_trace,
        "masks": masks,
    }
    if not args.smoke:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
        torch.save(checkpoint_payload, temporary)
        temporary.replace(args.checkpoint)
    if args.profile in RATE_DISTORTION_PROFILES:
        experiment_type = "layer_expert_rate_distortion_mask_generation"
    elif args.profile in REVIEW_PROFILES:
        experiment_type = "review_v1_fair_ablation_mask_generation"
    elif args.profile in UNIVERSAL_PROFILES:
        experiment_type = "cross_architecture_universal_mask_generation"
    else:
        experiment_type = "cross_architecture_camera_baseline_mask_generation"
    result = {
        "schema_version": 1,
        "experiment_type": experiment_type,
        "status": status,
        "smoke": args.smoke,
        "model": args.model_name,
        "model_revision": args.model_revision,
        "model_source": str(args.model_source.resolve()),
        "profile": args.profile,
        "method": profile.method,
        "allocation": "global",
        "prune_ratio": args.pruning_rate,
        "camera_alpha": args.camera_alpha,
        "anchor_general_weight": args.anchor_general_weight,
        "trust_radius_divisor": args.trust_radius_divisor,
        "interaction_strength": (
            profile.interaction_strength
            if args.interaction_strength_override is None
            else args.interaction_strength_override
        ),
        "fixed_stage1_checkpoint": (
            None
            if args.fixed_stage1_checkpoint is None
            else str(args.fixed_stage1_checkpoint.resolve())
        ),
        "fixed_stage1_checkpoint_sha256": fixed_stage1_checkpoint_sha256,
        "max_swaps": profile.max_swaps,
        "output_metric": (
            "next_rmsnorm_gamma" if args.profile != "camera" else None
        ),
        "calibration_corpus": profile.calibration_corpus,
        "calibration_aggregation": profile.calibration_aggregation,
        "layer_budget_schedule": profile.layer_budget_schedule,
        "layer_budgets": list(profile.layer_budgets),
        "rate_distortion_allocation": rate_distortion_trace,
        "rate_distortion_capacity_signal": profile.rate_distortion_capacity_signal,
        "rate_distortion_budget_scope": profile.rate_distortion_budget_scope,
        "rate_distortion_score_mode": profile.rate_distortion_score_mode,
        "rate_distortion_max_width_delta": profile.rate_distortion_max_width_delta,
        "tokens_per_expert": reservoir_size,
        "seed": SEED,
        "nsamples": nsamples,
        "sequence_length": sequence_length,
        "layers_completed": layer_limit,
        "model_layer_count": len(layers),
        "retained_channels": retained,
        "available_channels": available,
        "calibration": calibration,
        "layers": layer_summaries,
        "local_reconstruction": local_records,
        "mask_checkpoint": None if args.smoke else str(args.checkpoint.resolve()),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "command": sys.argv,
        "environment": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "model", "layers_completed", "retained_channels", "available_channels", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
