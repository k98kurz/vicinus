from vicinus.core import tokenize
from vicinus.errors import type_assert, value_assert


def jaccard_index(a: set[str], b: set[str]) -> float:
    """Calculate the Jaccard similarity index between two sets, e.g. of
        N-Grams. Defined as the cardinality of the intersection divided
        by the cardinality of the union.
    """
    usize = len(a.union(b))
    return len(a.intersection(b)) / usize if usize else 1.0

def n_grams(text: str, N: int = 3) -> set[str]:
    """Extract the N-Grams of a text, defined as the set of unique
        substrings of the given N.
    """
    type_assert(type(N) is int, 'N must be int')
    value_assert(N > 0, 'N must be positive')
    ng = set()
    text = ''.join(tokenize(text))
    for i in range(len(text)-N+1):
        ng.add(text[i:i+N])
    return ng
