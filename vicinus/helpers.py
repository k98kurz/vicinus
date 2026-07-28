from typing import Callable
from vicinus.errors import type_assert, value_assert


def rank(
        score_fn: Callable, candidates: list
    ) -> list[tuple[float, int]]:
    ranks = []
    for i in range(len(candidates)):
        ranks.append((score_fn(candidates[i]), i))

    ranks.sort(reverse=True)
    return ranks

def select(
        rankings: list[tuple[float, int]], k: int = 4
    ) -> list[tuple[float, int]]:
    """Select the top `k` from the rankings. Sorts beforehand. Assumes
        similarity scores and not distance scores. Raises `TypeError`
        or `ValueError` for invalid inputs.
    """
    type_assert(type(rankings) is list,
        'rankings must be list[tuple[float, int]]')
    type_assert(all([type(r) is tuple for r in rankings]),
        'rankings must be list[tuple[float, int]]')
    value_assert(all([len(r) == 2 for r in rankings]),
        'rankings must be list[tuple[float, int]]')
    type_assert(all([type(r[0]) is float and type(r[1]) is int for r in rankings]),
        'rankings must be list[tuple[float, int]]')
    type_assert(type(k) is int, 'k must be int>0')
    value_assert(k > 0, 'k must be int>0')
    rankings = sorted(rankings, reverse=True)
    return rankings[:k]
