from functools import lru_cache
from vicinus.errors import type_assert, value_assert


_head = lambda text: text[0]

def levenshtein_distance(a: str, b: str, normalize: bool = False) -> int|float:
    """Calculate Levenshtein distance between two strs. If `normalize`
        is `False` (default), distance is returned directly as an int.
        If `normalize` is set to `True`, the distance is normalized to a
        float proportional to the average length of the strings, clamped
        to a max of 1.0. Optimized with dynamic programming.
    """
    lal, lbl = len(a), len(b)
    if not lbl:
        return lal if not normalize else 1.0
    if not lal:
        return lbl if not normalize else 1.0

    state = [{}, {}]
    for i in range(lal):
        for j in range(lbl):
            if i == 0:
                state[1][j] = j
            elif j == 0:
                state[1][j] = i
            elif _head(a[i:]) == _head(b[j:]):
                state[1][j] = state[0][j-1]
            else:
                state[1][j] = 1 + min(
                    state[0][j],
                    state[1][j-1],
                    state[0][j-1]
                )
        state[0] = {**state[1]}

    d = state[0][lbl-1]
    return d if not normalize else d / max(lal, lbl)

def levenshtein_similarity(a: str, b: str) -> float:
    """Turns the normalized Levenshtein difference metric into a
        similarity metric by substracting it from 1.0.
    """
    return 1.0 - levenshtein_distance(a, b, True)
