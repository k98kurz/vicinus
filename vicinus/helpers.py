from vicinus.errors import type_assert, value_assert


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
    type_assert(all([type(r[0]) is float and type(r[1]) is int]),
        'rankings must be list[tuple[float, int]]')
    type_assert(type(k) is int, 'k must be int>0')
    value_assert(k > 0, 'k must be int>0')
    rankings = sorted(rankings, reverse=True)
    return rankings[:k]
