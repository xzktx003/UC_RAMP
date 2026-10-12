import pytest

from scripts.analyze_math_generation import (
    first_repeated_ngram_occurrence,
    prediction_diagnostic,
    repeated_ngram_fraction,
    summarize_diagnostics,
)


def _prediction(content: object, output_tokens: int, question_id: str) -> dict:
    return {
        "metadata": {"question_id": question_id},
        "model_output": {
            "choices": [{"message": {"content": content}}],
            "usage": {"output_tokens": output_tokens},
        },
    }


def test_repeated_ngram_fraction_counts_only_duplicate_occurrences() -> None:
    text = "a b c a b c a b c"

    assert repeated_ngram_fraction(text, n=3) == pytest.approx(4 / 7)
    assert repeated_ngram_fraction("a b", n=3) == 0.0


def test_first_repeated_ngram_occurrence_points_to_second_copy() -> None:
    text = "lead one two three gap one two three tail"

    occurrence = first_repeated_ngram_occurrence(text, n=3)

    assert occurrence is not None
    assert occurrence["ngram"] == ["one", "two", "three"]
    assert occurrence["first_word_index"] == 1
    assert occurrence["repeated_word_index"] == 5
    assert text[int(occurrence["repeated_character_start"]) :].startswith(
        "one two three"
    )


def test_first_repeated_ngram_occurrence_returns_none_without_duplicate() -> None:
    assert first_repeated_ngram_occurrence("one two three four", n=3) is None


def test_prediction_diagnostic_uses_fixed_cap_and_repetition_thresholds() -> None:
    record = _prediction("a b c a b c a b c", 2000, "q1")

    diagnostic = prediction_diagnostic(
        record, ngram_size=3, near_cap_tokens=2000, repetition_threshold=0.5
    )

    assert diagnostic["question_id"] == "q1"
    assert diagnostic["near_cap"] is True
    assert diagnostic["high_repetition"] is True


def test_prediction_diagnostic_accepts_openai_text_content_parts() -> None:
    record = _prediction(
        [
            {"type": "reasoning", "reasoning": ""},
            {"type": "text", "text": "a b c a b c a b c"},
        ],
        2000,
        "q-parts",
    )

    diagnostic = prediction_diagnostic(
        record, ngram_size=3, near_cap_tokens=2000, repetition_threshold=0.5
    )

    assert diagnostic["question_id"] == "q-parts"
    assert diagnostic["high_repetition"] is True


def test_summary_reports_rates_and_means() -> None:
    diagnostics = [
        {
            "output_tokens": 2000,
            "near_cap": True,
            "repeated_ngram_fraction": 0.5,
            "high_repetition": True,
        },
        {
            "output_tokens": 100,
            "near_cap": False,
            "repeated_ngram_fraction": 0.0,
            "high_repetition": False,
        },
    ]

    assert summarize_diagnostics(diagnostics) == {
        "num_samples": 2,
        "near_cap_rate": 0.5,
        "high_repetition_rate": 0.5,
        "mean_output_tokens": 1050.0,
        "mean_repeated_ngram_fraction": 0.25,
    }
