"""RAMP-MoE research implementation."""

from .core import (
    build_centered_difference_interaction_gram,
    build_energy_calibrated_difference_interaction_gram,
    build_explicit_gram,
    build_interaction_gram,
    diagonal_keep_indices,
    direct_reconstruction_error,
    forward_residual_keep_indices,
    gram_reconstruction_error,
    joint_mask_weights,
    minimum_norm_simplex_gradient,
    pareto_swap_refine_keep_indices,
    project_pareto_taylor_swap_prefix,
    reweight_gram_with_importance,
    slice_weights,
    swap_refine_keep_indices,
    swiglu_output,
)
from .selection import apply_qwen3_keep_mask_, select_camera_keep_mask
from .taylor import differentiable_qwen3_channel_gates

__all__ = [
    "build_centered_difference_interaction_gram",
    "build_energy_calibrated_difference_interaction_gram",
    "build_explicit_gram",
    "build_interaction_gram",
    "diagonal_keep_indices",
    "direct_reconstruction_error",
    "forward_residual_keep_indices",
    "gram_reconstruction_error",
    "joint_mask_weights",
    "minimum_norm_simplex_gradient",
    "pareto_swap_refine_keep_indices",
    "project_pareto_taylor_swap_prefix",
    "reweight_gram_with_importance",
    "slice_weights",
    "swap_refine_keep_indices",
    "swiglu_output",
    "apply_qwen3_keep_mask_",
    "select_camera_keep_mask",
    "differentiable_qwen3_channel_gates",
]
