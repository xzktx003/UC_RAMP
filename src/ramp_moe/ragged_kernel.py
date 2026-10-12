"""Two-stage Triton decode kernel for physically compact SwiGLU experts.

The first launch computes only retained intermediate channels; the second
reduces those channels into expert outputs.  Inputs are routed rows, not token
rows, so Top-K aggregation remains the serving engine's responsibility.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .compact_experts import CompactExperts


@triton.jit
def _ragged_gate_up(
    X, Expert, Gate, Up, Offsets, Activation,
    H: tl.constexpr, MAX_WIDTH: tl.constexpr,
    ROUTES_PER_TOKEN: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hidden = tl.arange(0, BLOCK_H)
    expert = tl.load(Expert + row)
    start = tl.load(Offsets + expert)
    width = tl.load(Offsets + expert + 1) - start
    token = row // ROUTES_PER_TOKEN
    x = tl.load(X + token * H + hidden, hidden < H, other=0).to(tl.float32)
    gate = tl.load(
        Gate + (start + channel[:, None]) * H + hidden[None, :],
        (channel[:, None] < width) & (hidden[None, :] < H), other=0,
    ).to(tl.float32)
    up = tl.load(
        Up + (start + channel[:, None]) * H + hidden[None, :],
        (channel[:, None] < width) & (hidden[None, :] < H), other=0,
    ).to(tl.float32)
    g = tl.sum(gate * x[None, :], axis=1).to(X.dtype.element_ty).to(tl.float32)
    u = tl.sum(up * x[None, :], axis=1).to(X.dtype.element_ty).to(tl.float32)
    silu = (g * tl.sigmoid(g)).to(X.dtype.element_ty).to(tl.float32)
    a = (silu * u).to(X.dtype.element_ty)
    tl.store(Activation + row * MAX_WIDTH + channel, a, channel < MAX_WIDTH)


@triton.jit
def _ragged_down(
    Expert, Down, Offsets, Activation, Weights, Output,
    H: tl.constexpr, MAX_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0)
    hidden = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    channels = tl.arange(0, BLOCK_C)
    expert = tl.load(Expert + row)
    start = tl.load(Offsets + expert)
    width = tl.load(Offsets + expert + 1) - start
    acc = tl.full((BLOCK_C, BLOCK_H), 0, tl.float32)
    for base in range(0, tl.cdiv(width, BLOCK_C)):
        channel = base * BLOCK_C + channels
        a = tl.load(
            Activation + row * MAX_WIDTH + channel,
            channel < width, other=0,
        ).to(tl.float32)
        down = tl.load(
            Down + (start + channel[:, None]) * H + hidden[None, :],
            (channel[:, None] < width) & (hidden[None, :] < H), other=0,
        ).to(tl.float32)
        acc += a[:, None] * down
    output = tl.sum(acc, axis=0).to(Output.dtype.element_ty).to(tl.float32)
    if HAS_WEIGHTS:
        output *= tl.load(Weights + row).to(tl.float32)
    tl.store(Output + row * H + hidden, output, hidden < H)


def ragged_expert_forward(
    compact: CompactExperts,
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    *,
    validate_indices: bool = False,
    routes_per_token: int = 1,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate routed rows with compact weights, without pruned-channel FLOPs.

    This is a decode-oriented kernel path.  Router-produced expert IDs are
    trusted on the hot path; set ``validate_indices=True`` for untrusted input.
    Benchmark it against the actual serving backend before asserting an
    end-to-end speedup.
    """

    if not hidden_states.is_cuda:
        raise ValueError("ragged kernel requires CUDA tensors")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != compact.hidden_size:
        raise ValueError("hidden_states must have shape [routed_rows, hidden_size]")
    if routes_per_token <= 0:
        raise ValueError("routes_per_token must be positive")
    routed_rows = hidden_states.shape[0] * routes_per_token
    if expert_indices.shape != (routed_rows,):
        raise ValueError("expert_indices must have one entry per routed row")
    if expert_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_indices must have integer dtype")
    if router_weights is not None and router_weights.shape != expert_indices.shape:
        raise ValueError("router_weights must have one entry per routed row")
    if any(t.device != hidden_states.device for t in (
        expert_indices, compact.gate, compact.up, compact.down, compact.offsets,
    )) or (router_weights is not None and router_weights.device != hidden_states.device):
        raise ValueError("inputs and compact weights must share a CUDA device")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if validate_indices and expert_indices.numel() and (
        torch.any(expert_indices < 0) or torch.any(expert_indices >= compact.expert_count)
    ):
        raise ValueError("expert index out of range")
    rows, hidden = routed_rows, hidden_states.shape[1]
    if out is not None and (
        out.shape != (rows, hidden) or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous [routed_rows, hidden_size]")
    output = out if out is not None else torch.empty(
        (rows, hidden), device=hidden_states.device, dtype=hidden_states.dtype
    )
    if rows == 0:
        return output
    max_width = compact.max_width
    if max_width == 0:
        return output.zero_()
    if workspace is not None and (
        workspace.shape != (rows, max_width)
        or workspace.dtype != hidden_states.dtype
        or workspace.device != hidden_states.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be contiguous [routed_rows, max_width]")
    activations = workspace if workspace is not None else torch.empty(
        (rows, max_width), device=hidden_states.device, dtype=hidden_states.dtype
    )
    _ragged_gate_up[(rows, triton.cdiv(max_width, 16))](
        hidden_states, expert_indices, compact.gate, compact.up,
        compact.offsets, activations,
        hidden, max_width, routes_per_token, 16, triton.next_power_of_2(hidden),
        num_warps=4,
    )
    _ragged_down[(rows, triton.cdiv(hidden, 64))](
        expert_indices, compact.down, compact.offsets, activations,
        router_weights if router_weights is not None else hidden_states,
        output, hidden, max_width, 32, 64,
        router_weights is not None, num_warps=4,
    )
    return output
