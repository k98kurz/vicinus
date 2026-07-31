from typing import Any, Callable
from vicinus.errors import type_assert, value_assert


def rank(
        score_fn: Callable, candidates: dict
    ) -> list[tuple[float, Any]]:
    """Ranks `candidates` using the `score_fn` callable. Returns a list
        of tuples in form `(score: float, index: int)`, sorted by score
        descending.
    """
    type_assert(type(candidates) is dict, 'candidates must be dict')
    ranks = []
    for k, v in candidates.items():
        ranks.append((score_fn(v), k))

    ranks.sort(reverse=True)
    return ranks


def select(
        rankings: list[tuple[float, Any]], k: int = 4
    ) -> list[tuple[float, int]]:
    """Select the top `k` from the rankings. Sorts beforehand. Assumes
        similarity scores and not distance scores. Raises `TypeError`
        or `ValueError` for invalid inputs.
    """
    type_assert(type(rankings) is list,
        'rankings must be list[tuple[float, Any]]')
    type_assert(all([type(r) is tuple for r in rankings]),
        'rankings must be list[tuple[float, Any]]')
    value_assert(all([len(r) == 2 for r in rankings]),
        'rankings must be list[tuple[float, Any]]')
    type_assert(all([type(r[0]) is float for r in rankings]),
        'rankings must be list[tuple[float, Any]]')
    type_assert(type(k) is int, 'k must be int>0')
    value_assert(k > 0, 'k must be int>0')
    rankings = sorted(rankings, reverse=True)
    return rankings[:k]


def tokenize(text: str) -> list[str]:
    """Preprocess some text by casting to lower, removing punctuation,
        and splitting on word boundaries. Returns only alphanumeric
        tokens.
    """
    return [
        ''.join(c for c in word if c.isalnum())
        for word in text.lower().split()
    ]
