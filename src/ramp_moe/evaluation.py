"""Evaluation protocol objects with explicit smoke/formal separation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator, Sequence
import math

from datasets import DownloadConfig
import torch
import torch.nn.functional as F


def dataset_download_config(local_files_only: bool) -> DownloadConfig | None:
    """Keep cached dataset reads offline when the run forbids downloads."""

    if not local_files_only:
        return None
    return DownloadConfig(local_files_only=True)


@dataclass(frozen=True)
class WikiTextPPLConfig:
    """Immutable WikiText-2 PPL protocol configuration."""

    seq_len: int = 2048
    smoke: bool = False
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than one")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        if not self.smoke and self.seq_len != 2048:
            raise ValueError("formal WikiText-2 evaluation requires seq_len=2048")
        if not self.smoke and self.max_windows is not None:
            raise ValueError(
                "formal WikiText-2 evaluation cannot limit the number of windows"
            )

    @property
    def result_label(self) -> str:
        return "smoke" if self.smoke else "formal"


def iter_token_windows(
    input_ids: torch.Tensor, config: WikiTextPPLConfig
) -> Iterator[torch.Tensor]:
    """Yield complete, non-overlapping token windows under the protocol."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, token_count]")
    available = input_ids.shape[1] // config.seq_len
    window_count = available
    if config.max_windows is not None:
        window_count = min(window_count, config.max_windows)
    for index in range(window_count):
        start = index * config.seq_len
        yield input_ids[:, start : start + config.seq_len]


def perplexity_from_nll(
    window_nll_sums: Sequence[float], predicted_token_counts: Sequence[int]
) -> float:
    """Aggregate window NLL sums using their actual prediction counts."""

    if len(window_nll_sums) != len(predicted_token_counts):
        raise ValueError("NLL and token-count sequences must have equal length")
    if not window_nll_sums:
        raise ValueError("at least one evaluated window is required")
    if any(count <= 0 for count in predicted_token_counts):
        raise ValueError("predicted token counts must be positive")
    return math.exp(sum(window_nll_sums) / sum(predicted_token_counts))


@torch.no_grad()
def evaluate_causal_lm_ppl(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config: WikiTextPPLConfig,
    *,
    input_device: torch.device | str,
) -> dict[str, object]:
    """Evaluate a causal LM without vLLM on complete WikiText-style windows."""

    model.eval()
    nll_sums: list[float] = []
    predicted_counts: list[int] = []
    for window in iter_token_windows(input_ids, config):
        window = window.to(input_device)
        logits = model(input_ids=window, use_cache=False).logits
        shifted_logits = logits[:, :-1].float().contiguous()
        shifted_labels = window[:, 1:].contiguous()
        nll = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            reduction="sum",
        )
        nll_sums.append(float(nll.item()))
        predicted_counts.append(int(shifted_labels.numel()))

    return {
        "ppl": perplexity_from_nll(nll_sums, predicted_counts),
        "window_nll_sums": nll_sums,
        "predicted_token_counts": predicted_counts,
        "num_windows": len(nll_sums),
        "num_predicted_tokens": sum(predicted_counts),
        "seq_len": config.seq_len,
        "result_label": config.result_label,
        "smoke": config.smoke,
    }
