"""Physically compact, variable-width SwiGLU expert weights.

This representation is independent of a serving engine.  It keeps the exact
selected gate/up rows and matching down columns, so a kernel can use per-expert
offsets without executing pruned channels.  ``forward_reference`` is a
correctness oracle, not a latency-optimized inference path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .moe_adapter import MoELayerView


@dataclass(frozen=True)
class CompactExperts:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    offsets: torch.Tensor
    hidden_size: int
    max_width: int

    @property
    def expert_count(self) -> int:
        return self.offsets.numel() - 1

    @property
    def widths(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        router_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate routed token/expert pairs without allocating pruned channels.

        Each row is one already-routed token.  Duplicate token rows may be
        supplied for Top-K routing; the caller aggregates their outputs.
        """

        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [routed_rows, hidden_size]")
        if expert_indices.shape != (hidden_states.shape[0],):
            raise ValueError("expert_indices must have one entry per routed row")
        if expert_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_indices must have integer dtype")
        if router_weights is not None and router_weights.shape != expert_indices.shape:
            raise ValueError("router_weights must have one entry per routed row")
        if hidden_states.device != self.gate.device or expert_indices.device != self.gate.device:
            raise ValueError("inputs and compact weights must be on the same device")
        if expert_indices.numel() and (
            torch.any(expert_indices < 0) or torch.any(expert_indices >= self.expert_count)
        ):
            raise ValueError("expert index out of range")

        result = torch.zeros_like(hidden_states)
        for expert_index in range(self.expert_count):
            rows = torch.where(expert_indices == expert_index)[0]
            if rows.numel() == 0:
                continue
            start = int(self.offsets[expert_index])
            stop = int(self.offsets[expert_index + 1])
            if start == stop:
                continue
            x = hidden_states.index_select(0, rows)
            activation = F.silu(x @ self.gate[start:stop].T) * (
                x @ self.up[start:stop].T
            )
            output = activation @ self.down[start:stop]
            if router_weights is not None:
                output = output * router_weights.index_select(0, rows).unsqueeze(1)
            result.index_copy_(0, rows, output)
        return result


@torch.no_grad()
def pack_compact_experts(moe: MoELayerView, keep_mask: torch.Tensor) -> CompactExperts:
    """Pack coupled gate/up/down channels according to an explicit mask."""

    if keep_mask.dtype != torch.bool or keep_mask.shape != (
        moe.expert_count,
        moe.channel_count,
    ):
        raise ValueError("keep_mask must be boolean [expert_count, channel_count]")

    gates: list[torch.Tensor] = []
    ups: list[torch.Tensor] = []
    downs: list[torch.Tensor] = []
    offsets = [0]
    for expert_index in range(moe.expert_count):
        selected = torch.where(keep_mask[expert_index])[0]
        if moe.packed:
            gate_up = moe.experts.gate_up_proj[expert_index]
            down = moe.experts.down_proj[expert_index]
            selected = selected.to(gate_up.device)
            gate = gate_up[: moe.channel_count].index_select(0, selected)
            up = gate_up[moe.channel_count :].index_select(0, selected)
            down_rows = down.index_select(1, selected).T
        else:
            expert = moe.experts[expert_index]
            gate = expert.gate_proj.weight.index_select(
                0, selected.to(expert.gate_proj.weight.device)
            )
            up = expert.up_proj.weight.index_select(
                0, selected.to(expert.up_proj.weight.device)
            )
            down_rows = expert.down_proj.weight.index_select(
                1, selected.to(expert.down_proj.weight.device)
            ).T
        if gate.dtype != up.dtype or gate.dtype != down_rows.dtype:
            raise ValueError("expert projections must share a dtype")
        if gate.device != up.device or gate.device != down_rows.device:
            raise ValueError("expert projections must share a device")
        gates.append(gate.contiguous())
        ups.append(up.contiguous())
        downs.append(down_rows.contiguous())
        offsets.append(offsets[-1] + selected.numel())

    device = gates[0].device
    return CompactExperts(
        gate=torch.cat(gates, dim=0),
        up=torch.cat(ups, dim=0),
        down=torch.cat(downs, dim=0),
        offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
        hidden_size=moe.hidden_size,
        max_width=max(offsets[index + 1] - offsets[index] for index in range(moe.expert_count)),
    )
