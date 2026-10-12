"""Deterministic text-generation pathology primitives."""

from __future__ import annotations

from collections import Counter
import re


def repeated_ngram_fraction(text: str, *, n: int = 8) -> float:
    """Return the fraction of whitespace n-grams beyond their first occurrence."""

    if n <= 0:
        raise ValueError("n must be positive")
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    duplicate_occurrences = sum(count - 1 for count in Counter(ngrams).values())
    return duplicate_occurrences / len(ngrams)


def first_repeated_ngram_occurrence(
    text: str, *, n: int = 8
) -> dict[str, object] | None:
    """Locate the first n-gram occurrence that repeats an earlier n-gram."""

    if n <= 0:
        raise ValueError("n must be positive")
    matches = list(re.finditer(r"\S+", text))
    if len(matches) < n:
        return None
    words = [match.group(0) for match in matches]
    first_indices: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - n + 1):
        ngram = tuple(words[index : index + n])
        first_index = first_indices.get(ngram)
        if first_index is None:
            first_indices[ngram] = index
            continue
        return {
            "ngram": list(ngram),
            "first_word_index": first_index,
            "repeated_word_index": index,
            "repeated_character_start": matches[index].start(),
            "repeated_character_end": matches[index + n - 1].end(),
        }
    return None
