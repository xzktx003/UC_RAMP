"""Local vLLM adapter for physically compact Qwen3 routed experts.

The adapter is intentionally kept outside site-packages.  It provides a custom
vLLM Qwen3 architecture that installs compact routed experts inside the worker
after checkpoint weights are loaded.

Scope of the first integration:
- unquantized Qwen3 MoE routed experts with canonical vLLM tensors;
- non-EPLB layouts where global expert ids map directly;
- CUDA execution through this repository's ragged kernel, with a CPU reference
  path for tests.

Unsupported layouts fail closed before replacing modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Iterable

import torch
from torch import nn

from .compact_experts import CompactExperts
from .ragged_kernel import ragged_expert_forward

COMPACT_QWEN3_ARCH = "Qwen3MoeCompactForCausalLM"
MASK_CHECKPOINT_ENV = "RAMP_MOE_VLLM_MASK_CHECKPOINT"
AUTO_REGISTER_ENV = "RAMP_MOE_VLLM_AUTO_REGISTER"
DEBUG_MEMORY_ENV = "RAMP_MOE_VLLM_DEBUG_MEMORY"
VALIDATE_INDICES_ENV = "RAMP_MOE_VLLM_VALIDATE_INDICES"


@dataclass(frozen=True)
class VllmCompactInstallReport:
    """Summary of an in-place vLLM compact expert installation."""

    layer_count: int
    widths_by_layer: tuple[tuple[int, ...], ...]
    retained_channels: int
    original_channels: int
    memory_telemetry: tuple[dict[str, object], ...] = ()


def register_compact_qwen3_model() -> None:
    """Register the worker-side compact Qwen3 vLLM architecture."""

    try:
        from vllm import ModelRegistry
    except ModuleNotFoundError as exc:
        raise RuntimeError("vLLM is required to register the compact Qwen3 model") from exc
    ModelRegistry.register_model(
        COMPACT_QWEN3_ARCH,
        "ramp_moe.vllm_compact:Qwen3MoeCompactForCausalLM",
    )


class CompactModelRegistryWorkerExtension:
    """Import-time hook for vLLM workers.

    vLLM resolves ``worker_extension_cls`` during worker initialization.  The
    class intentionally has no public methods; importing this module performs
    registration when ``RAMP_MOE_VLLM_AUTO_REGISTER=1`` is present.
    """

    pass


def _linear_output(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call vLLM/HF linear modules and normalize tuple-returning APIs."""

    value = module(x)
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("router/gate module must return a tensor or (tensor, ...)")
    return value


def _validate_keep_mask(keep_mask: torch.Tensor, expert_count: int, width: int) -> None:
    if keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must have boolean dtype")
    if keep_mask.shape != (expert_count, width):
        raise ValueError(f"keep_mask must have shape {(expert_count, width)}")


def _masks_from_object(value: object) -> list[torch.Tensor]:
    if isinstance(value, list) and all(isinstance(mask, torch.Tensor) for mask in value):
        return value
    if isinstance(value, tuple) and all(isinstance(mask, torch.Tensor) for mask in value):
        return list(value)
    if isinstance(value, dict) and all(isinstance(mask, torch.Tensor) for mask in value.values()):
        def key_order(key: object) -> tuple[int, object]:
            text = str(key)
            return (0, int(text)) if text.isdigit() else (1, text)

        return [value[key] for key in sorted(value, key=key_order)]
    raise ValueError("mask data must be a list/tuple of tensors or a dict of tensors")


def load_masks_from_checkpoint(path: str | Path) -> list[torch.Tensor]:
    """Load keep masks from this repository's pruning checkpoint schemas."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("masks", "keep_masks", "mask"):
            if key in checkpoint:
                return _masks_from_object(checkpoint[key])
    return _masks_from_object(checkpoint)


@torch.no_grad()
def pack_vllm_routed_experts(
    routed_experts: nn.Module,
    keep_mask: torch.Tensor,
    *,
    output_device: torch.device | str | None = None,
) -> CompactExperts:
    """Pack an unquantized vLLM ``RoutedExperts`` object.

    vLLM's unquantized Qwen3 routed experts expose:
    ``w13_weight: [experts, 2 * channels, hidden]`` and
    ``w2_weight: [experts, hidden, channels]`` before any exotic quantized
    layout is considered.  This function rejects tensors that do not match
    that contract rather than guessing.
    """

    if getattr(routed_experts, "quant_config", None) is not None:
        raise NotImplementedError("compact vLLM packing currently supports unquantized experts only")
    expert_map = getattr(routed_experts, "expert_map", None)
    if isinstance(expert_map, torch.Tensor):
        expected = torch.arange(expert_map.numel(), device=expert_map.device)
        if not torch.equal(expert_map.to(torch.long), expected):
            raise NotImplementedError(
                "compact vLLM packing requires a direct global-to-local expert map"
            )

    w13 = getattr(routed_experts, "w13_weight", None)
    w2 = getattr(routed_experts, "w2_weight", None)
    if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
        raise TypeError("vLLM routed experts must expose w13_weight and w2_weight")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("vLLM routed expert weights must be rank-three tensors")
    expert_count, doubled_width, hidden_size = w13.shape
    if doubled_width % 2:
        raise ValueError("w13_weight second dimension must be 2 * channel_count")
    channel_count = doubled_width // 2
    if w2.shape != (expert_count, hidden_size, channel_count):
        raise ValueError(
            "w2_weight must have shape [experts, hidden_size, channel_count]"
        )
    if w13.dtype != w2.dtype or w13.device != w2.device:
        raise ValueError("w13_weight and w2_weight must share dtype and device")
    _validate_keep_mask(keep_mask, expert_count, channel_count)

    target_device = torch.device(output_device) if output_device is not None else w13.device
    widths = keep_mask.sum(dim=1, dtype=torch.int64).cpu()
    offsets_cpu = torch.empty(expert_count + 1, dtype=torch.int64)
    offsets_cpu[0] = 0
    offsets_cpu[1:] = torch.cumsum(widths, dim=0)
    total_width = int(offsets_cpu[-1].item())

    gate = torch.empty((total_width, hidden_size), dtype=w13.dtype, device=target_device)
    up = torch.empty_like(gate)
    down = torch.empty_like(gate)

    for expert_index in range(expert_count):
        start = int(offsets_cpu[expert_index].item())
        end = int(offsets_cpu[expert_index + 1].item())
        if start == end:
            continue
        selected = keep_mask[expert_index].nonzero().flatten().to(w13.device)
        gate[start:end].copy_(
            w13[expert_index, :channel_count]
            .index_select(0, selected)
            .to(target_device)
        )
        up[start:end].copy_(
            w13[expert_index, channel_count:]
            .index_select(0, selected)
            .to(target_device)
        )
        down[start:end].copy_(
            w2[expert_index].index_select(1, selected).T.to(target_device)
        )

    return CompactExperts(
        gate=gate.contiguous(),
        up=up.contiguous(),
        down=down.contiguous(),
        offsets=offsets_cpu.to(device=target_device),
        hidden_size=hidden_size,
        max_width=int(widths.max().item()) if expert_count else 0,
    )


class CompactVllmQwen3Experts(nn.Module):
    """Replacement for vLLM Qwen3 ``block.experts``.

    It keeps vLLM's block-level call shape:
    ``experts(hidden_states=..., router_logits=...)``.  The router logits
    argument is ignored because Qwen3's vLLM runner receives hidden states and
    applies the block's ``gate`` internally.
    """

    def __init__(
        self,
        compact: CompactExperts,
        *,
        gate: nn.Module,
        top_k: int,
        renormalize: bool,
        shared_expert: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.gate_module = gate
        self.shared_expert = shared_expert
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.register_buffer("gate", compact.gate)
        self.register_buffer("up", compact.up)
        self.register_buffer("down", compact.down)
        self.register_buffer("offsets", compact.offsets)
        self.hidden_size = compact.hidden_size
        self.max_width = compact.max_width

    @property
    def compact(self) -> CompactExperts:
        return CompactExperts(
            self.gate,
            self.up,
            self.down,
            self.offsets,
            self.hidden_size,
            self.max_width,
        )

    @property
    def is_internal_router(self) -> bool:
        return True

    @property
    def shared_experts(self) -> nn.Module | None:
        return self.shared_expert

    def update_expert_map(self) -> None:
        """Fail closed for vLLM EPLB/EP metadata updates.

        Compact experts are packed under a direct global expert layout.  vLLM's
        metadata update path is designed for dense routed runners and would not
        know how to reshuffle ragged compact buffers.
        """

        raise NotImplementedError(
            "compact vLLM Qwen3 experts do not support EPLB/EP expert-map updates"
        )

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = _linear_output(self.gate_module, hidden_states)
        if logits.ndim != 2:
            raise ValueError("router logits must have shape [tokens, experts]")
        if not 0 < self.top_k <= logits.shape[1]:
            raise ValueError("top_k must lie inside the expert count")
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, k=self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return ids.to(torch.int64), weights.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        topk_ids, topk_weights = self._route(hidden_states)
        routed_ids = topk_ids.reshape(-1).contiguous()
        routed_weights = topk_weights.reshape(-1).contiguous()
        if hidden_states.is_cuda:
            routed_output = ragged_expert_forward(
                self.compact,
                hidden_states,
                routed_ids,
                routed_weights,
                routes_per_token=self.top_k,
                validate_indices=os.environ.get(VALIDATE_INDICES_ENV) == "1",
            )
        else:
            routed_input = hidden_states.repeat_interleave(self.top_k, dim=0)
            routed_output = self.compact.forward_reference(
                routed_input,
                routed_ids,
                routed_weights,
            )
        output = routed_output.view(hidden_states.shape[0], self.top_k, -1).sum(dim=1)
        if self.shared_expert is not None:
            output = output + self.shared_expert(hidden_states)
        return output


def _decoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    for path in (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("llm", "model", "layers"),
    ):
        value: object = model
        for name in path:
            if not hasattr(value, name):
                value = None
                break
            value = getattr(value, name)
        if isinstance(value, (nn.ModuleList, list, tuple)):
            return value
    raise ValueError("unable to locate vLLM Qwen3 decoder layers")


def _routed_experts_from_runner(experts: nn.Module) -> nn.Module:
    routed = getattr(experts, "routed_experts", None)
    if isinstance(routed, nn.Module):
        return routed
    return experts


def _runner_top_k(runner: nn.Module, block: nn.Module, routed_experts: nn.Module) -> int:
    for candidate in (
        getattr(runner, "top_k", None),
        getattr(getattr(runner, "moe_config", None), "experts_per_token", None),
        getattr(getattr(routed_experts, "moe_config", None), "experts_per_token", None),
        getattr(block, "top_k", None),
    ):
        if candidate is not None:
            return int(candidate)
    raise ValueError("unable to infer top_k from vLLM MoE runner")


def _runner_renormalize(runner: nn.Module, routed_experts: nn.Module) -> bool:
    for owner in (
        runner,
        getattr(runner, "router", None),
        routed_experts,
    ):
        value = getattr(owner, "renormalize", None)
        if value is not None:
            return bool(value)
    return True


def _install_metadata_fail_closed_(model: nn.Module) -> None:
    if not hasattr(model, "update_physical_experts_metadata"):
        return

    def update_physical_experts_metadata_unsupported(
        self: nn.Module,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        raise NotImplementedError(
            "compact vLLM Qwen3 serving does not support changing physical "
            "expert metadata after packing; disable EPLB/redundant expert "
            "metadata updates for this path"
        )

    model.update_physical_experts_metadata = MethodType(
        update_physical_experts_metadata_unsupported,
        model,
    )


def _replace_moe_layer_entry_(
    model: nn.Module,
    index: int,
    value: nn.Module | None,
) -> bool:
    moe_layers = getattr(model, "moe_layers", None)
    if not isinstance(moe_layers, (list, nn.ModuleList)):
        return False
    if index >= len(moe_layers):
        return False
    moe_layers[index] = value
    return True


def _maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()


def _debug_cuda_memory(
    telemetry: list[dict[str, object]],
    *,
    layer_index: int,
    stage: str,
    device: torch.device,
) -> None:
    if os.environ.get(DEBUG_MEMORY_ENV) != "1" or device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    record: dict[str, object] = {
        "event": "ramp_moe_vllm_compact_memory",
        "layer_index": layer_index,
        "stage": stage,
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    telemetry.append(record)
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def _compact_to_device(compact: CompactExperts, device: torch.device) -> CompactExperts:
    return CompactExperts(
        gate=compact.gate.to(device=device),
        up=compact.up.to(device=device),
        down=compact.down.to(device=device),
        offsets=compact.offsets.to(device=device),
        hidden_size=compact.hidden_size,
        max_width=compact.max_width,
    )


def _release_module_tensor_storage_(module: nn.Module) -> None:
    """Move obsolete module-owned tensors to empty CPU storage in place.

    During vLLM weight loading, external Python objects may still hold old
    routed-expert modules or their Parameters.  Detaching the module tree is
    therefore not sufficient to release CUDA storage.  After compact CPU
    buffers have been materialized, this helper makes the obsolete routed
    expert harmless even if such stale Python references remain alive.
    """

    for parameter in module.parameters(recurse=True):
        parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
    for buffer in module.buffers(recurse=True):
        buffer.data = torch.empty(0, dtype=buffer.dtype, device="cpu")


@torch.no_grad()
def install_compact_qwen3_moe_(
    model: nn.Module,
    masks: list[torch.Tensor],
) -> VllmCompactInstallReport:
    """Replace vLLM Qwen3 routed experts in place with compact modules.

    ``model`` should be the loaded vLLM model object, not a HuggingFace model.
    The operation is deliberately strict: if a layer does not look like Qwen3's
    sparse block or if its expert weights are quantized/reformatted, no partial
    replacement is attempted for that layer.
    """

    layers = list(_decoder_layers(model))
    sparse_blocks: list[nn.Module] = []
    for layer in layers:
        block = getattr(layer, "mlp", None)
        if (
            isinstance(block, nn.Module)
            and hasattr(block, "gate")
            and hasattr(block, "experts")
            and hasattr(block, "n_routed_experts")
        ):
            sparse_blocks.append(block)
    if len(sparse_blocks) != len(masks):
        raise ValueError(
            f"expected {len(sparse_blocks)} masks for vLLM sparse layers, got {len(masks)}"
        )

    widths_by_layer: list[tuple[int, ...]] = []
    compact_layers: list[CompactVllmQwen3Experts] = []
    memory_telemetry: list[dict[str, object]] = []
    retained = 0
    original = 0
    for layer_index, (block, mask) in enumerate(zip(sparse_blocks, masks)):
        runner = getattr(block, "experts")
        routed_experts = _routed_experts_from_runner(runner)
        w13 = getattr(routed_experts, "w13_weight", None)
        weight_device = w13.device if isinstance(w13, torch.Tensor) else torch.device("cpu")
        top_k = _runner_top_k(runner, block, routed_experts)
        renormalize = _runner_renormalize(runner, routed_experts)
        shared_expert = getattr(block, "shared_expert", None)
        _debug_cuda_memory(
            memory_telemetry,
            layer_index=layer_index,
            stage="before_pack",
            device=weight_device,
        )
        block.experts = None
        _replace_moe_layer_entry_(model, layer_index, None)
        compact_cpu = pack_vllm_routed_experts(
            routed_experts,
            mask.to("cpu"),
            output_device=torch.device("cpu"),
        )
        _release_module_tensor_storage_(routed_experts)
        if getattr(runner, "routed_experts", None) is routed_experts:
            runner.routed_experts = None
        del runner, routed_experts, w13
        gc.collect()
        _maybe_empty_cuda_cache(weight_device)
        _debug_cuda_memory(
            memory_telemetry,
            layer_index=layer_index,
            stage="after_drop_old",
            device=weight_device,
        )
        compact = _compact_to_device(compact_cpu, weight_device)
        _debug_cuda_memory(
            memory_telemetry,
            layer_index=layer_index,
            stage="after_upload",
            device=weight_device,
        )
        del compact_cpu
        compact_experts = CompactVllmQwen3Experts(
            compact,
            gate=block.gate,
            top_k=int(top_k),
            renormalize=renormalize,
            shared_expert=shared_expert,
        )
        block.experts = compact_experts
        compact_layers.append(compact_experts)
        _replace_moe_layer_entry_(model, layer_index, compact_experts)
        widths = tuple(int(x) for x in mask.sum(dim=1).tolist())
        widths_by_layer.append(widths)
        retained += sum(widths)
        original += mask.numel()
        del compact
        _maybe_empty_cuda_cache(weight_device)
    if hasattr(model, "moe_layers"):
        moe_layers = getattr(model, "moe_layers")
        if not isinstance(moe_layers, (list, nn.ModuleList)) or len(moe_layers) != len(compact_layers):
            model.moe_layers = compact_layers
    _install_metadata_fail_closed_(model)

    return VllmCompactInstallReport(
        layer_count=len(widths_by_layer),
        widths_by_layer=tuple(widths_by_layer),
        retained_channels=retained,
        original_channels=original,
        memory_telemetry=tuple(memory_telemetry),
    )


try:
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeForCausalLM as _VllmQwen3MoeForCausalLM,
    )
except ModuleNotFoundError:
    _VllmQwen3MoeForCausalLM = None


if _VllmQwen3MoeForCausalLM is not None:

    class Qwen3MoeCompactForCausalLM(_VllmQwen3MoeForCausalLM):  # type: ignore[misc]
        """vLLM worker-side Qwen3 model that installs compact experts after load."""

        def load_weights(self, weights):  # type: ignore[no-untyped-def]
            loaded = super().load_weights(weights)
            mask_checkpoint = os.environ.get(MASK_CHECKPOINT_ENV)
            if not mask_checkpoint:
                raise RuntimeError(
                    f"{MASK_CHECKPOINT_ENV} must point to a local mask checkpoint "
                    f"when using {COMPACT_QWEN3_ARCH}"
                )
            masks = load_masks_from_checkpoint(mask_checkpoint)
            report = install_compact_qwen3_moe_(self, masks)
            self.ramp_moe_vllm_compact_report = report
            return loaded

else:

    class Qwen3MoeCompactForCausalLM(nn.Module):
        """Placeholder used only when vLLM is absent from the Python env."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("vLLM is required for Qwen3MoeCompactForCausalLM")


if os.environ.get(AUTO_REGISTER_ENV) == "1":
    register_compact_qwen3_model()
