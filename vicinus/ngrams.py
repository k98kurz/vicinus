from vicinus.errors import type_assert, value_assert
from vicinus.tokenize import tokenize


def jaccard_index(a: set, b: set) -> float:
    """Calculate the Jaccard similarity index between two sets, e.g. of
        N-Grams. Defined as the cardinality of the intersection divided
        by the cardinality of the union.
    """
    usize = len(a.union(b))
    return len(a.intersection(b)) / usize if usize else 1.0

def n_grams(text: str, size: int = 3) -> set[str]:
    """Extract the N-Grams of a text, defined as the set of unique
        substrings of the given size.
    """
    type_assert(type(size) is int, 'size must be int')
    value_assert(size > 0, 'size must be positive')
    ng = set()
    text = ''.join(tokenize(text))
    for i in range(len(text)-size+1):
        ng.add(text[i:i+size])
    return ng
