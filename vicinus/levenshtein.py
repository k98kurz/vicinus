from functools import lru_cache
from vicinus.errors import type_assert, value_assert


_head = lambda text: text[0]
_tail = lambda text: text[1:]

@lru_cache(maxsize=1024)
def levenshtein_distance(a: str, b: str, normalize: bool = False) -> int|float:
    """Calculate Levenshtein distance between two strs. If `normalize`
        is `False` (default), distance is returned directly as an int.
        If `normalize` is set to `True`, the distance is normalized to a
        float proportional to the average length of the strings, clamped
        to a max of 1.0. Optimized by `@lru_cache` decorator -- be sure
        to use `.lower()` or `' '.join(tokenize())` on strings before
        passing them in to avoid cache fragmentation.
    """
    lal, lbl = len(a), len(b)
    if not lbl:
        return lal if not normalize else 1.0
    if not lal:
        return lbl if not normalize else 1.0
    if _head(a) == _head(b):
        d = levenshtein_distance(_tail(a), _tail(b))
        return d if not normalize else d / max(lal, lbl)
    d = 1 + min(
        levenshtein_distance(_tail(a), b),
        levenshtein_distance(a, _tail(b)),
        levenshtein_distance(_tail(a), _tail(b))
    )
    return d if not normalize else d / max(lal, lbl)

def levenshtein_similarity(a: str, b: str) -> float:
    """Turns the normalized Levenshtein difference metric into a
        similarity metric by substracting it from 1.0.
    """
    return 1.0 - levenshtein_distance(a, b, True)
