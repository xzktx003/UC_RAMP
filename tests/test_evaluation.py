import math

import pytest
import torch

from ramp_moe.evaluation import (
    WikiTextPPLConfig,
    dataset_download_config,
    iter_token_windows,
    perplexity_from_nll,
)


def test_local_dataset_mode_disables_remote_downloads() -> None:
    config = dataset_download_config(True)

    assert config is not None
    assert config.local_files_only
    assert dataset_download_config(False) is None


def test_formal_wikitext_evaluation_cannot_limit_windows() -> None:
    with pytest.raises(ValueError, match="formal WikiText-2"):
        WikiTextPPLConfig(seq_len=2048, smoke=False, max_windows=2)


def test_smoke_wikitext_evaluation_is_explicitly_labeled() -> None:
    config = WikiTextPPLConfig(seq_len=2048, smoke=True, max_windows=2)

    assert config.result_label == "smoke"


def test_formal_wikitext_protocol_uses_2048_tokens() -> None:
    config = WikiTextPPLConfig()

    assert config.seq_len == 2048
    assert config.max_windows is None
    assert config.result_label == "formal"


def test_token_windows_drop_remainder_without_overlap() -> None:
    token_ids = torch.arange(11).unsqueeze(0)
    config = WikiTextPPLConfig(seq_len=4, smoke=True)

    windows = list(iter_token_windows(token_ids, config))

    assert [window.tolist() for window in windows] == [
        [[0, 1, 2, 3]],
        [[4, 5, 6, 7]],
    ]


def test_smoke_window_limit_is_applied() -> None:
    token_ids = torch.arange(12).unsqueeze(0)
    config = WikiTextPPLConfig(seq_len=4, smoke=True, max_windows=1)

    assert len(list(iter_token_windows(token_ids, config))) == 1


def test_perplexity_is_weighted_by_predicted_token_count() -> None:
    perplexity = perplexity_from_nll(
        window_nll_sums=[2.0, 6.0], predicted_token_counts=[1, 3]
    )

    assert perplexity == pytest.approx(math.exp(2.0))
