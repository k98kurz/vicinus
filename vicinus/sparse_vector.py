from __future__ import annotations
from typing import Iterator, ItemsView, ValuesView
from vicinus.errors import type_assert


class SparseVector:
    """Class representing a sparse vector, i.e. omitting every element
        that equals 0. Useful for high dimensionality vectors with
        mostly 0s.
    """
    def __init__(self, data: dict[int, int|float] = None):
        """Initialize. Raises `TypeError` for invalid data."""
        data = data or {}
        self._data = {}
        for k,v in data.items():
            self[k] = v

    def __getitem__(self, index: int) -> int|float:
        type_assert(isinstance(index, int), 'index must be int')
        return self._data.get(index, 0)

    def __setitem__(self, index: int, value: int|float) -> None:
        type_assert(type(index) is int, 'index must be int')
        type_assert(type(value) in (int, float), 'value must be int|float')
        if value != 0:
            self._data[index] = value
        elif index in self._data:
            del self._data[index]

    def __delitem__(self, index: int) -> None:
        type_assert(type(index) is int, 'index must be int')
        del self._data[index]

    def __iter__(self) -> Iterator:
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"SparseVector{repr(self._data)}"

    def get(self, index: int, default = 0) -> int|float|None:
        """Get the element at the specified index."""
        type_assert(isinstance(index, int), 'index must be int')
        return self._data.get(index, default)

    def keys(self) -> set[int]:
        """Return the set of indices for which elements exist."""
        return set(self._data.keys())

    def values(self) -> ValuesView[int|float]:
        """Return a view of all elements."""
        return self._data.values()

    def items(self) -> ItemsView[tuple[int, int|float]]:
        """Return an ItemsView for `for i, e in vec.items()` syntax."""
        return self._data.items()

    def norm(self) -> float:
        """Calculate the Euclidean norm of a SparseVector, defined as
            the square root of the dot product of itself (i.e. the sqrt
            of the sum of the squares of the elements). Raises
            `TypeError` for invalid input.
        """
        return self.dot_product(self) ** 0.5

    def dot_product(self, other: SparseVector) -> float:
        """Calculate the dot product between two `SparseVector`s. Raises
            `TypeError` for invalid other. Treats unset indices as 0
            values mathematically.
        """
        type_assert(isinstance(other, SparseVector), 'other must be SparseVector')
        # only the common indices are used; any missing index has value of 0
        indices = self.keys() & other.keys()
        return sum(self[i] * other[i] for i in indices)

    def cosine_similarity(self, other: SparseVector) -> float:
        """Calculate the cosine similarity with another SparseVector.
            Raises `TypeError`.
        """
        type_assert(isinstance(other, SparseVector), 'other must be SparseVector')
        n = self.norm() * other.norm()
        if n == 0:
            return 0.0
        cs = self.dot_product(other) / n
        # normalize to [-1.0, 1.0] to fix floating point precision issues
        return min(1.0, max(cs, -1.0))

    def copy(self) -> SparseVector:
        return SparseVector({**self._data})
