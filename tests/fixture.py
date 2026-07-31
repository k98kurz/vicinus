from dataclasses import dataclass, field
from vicinus import SparseVector, list_samples, get_sample
import os


@dataclass
class Document:
    title: str
    text: str
    vector: SparseVector = field(default=None)


class DocDB:
    def __init__(self):
        self._docs = []

        files = list_samples()
        for file in files:
            self._docs.append(Document(title=file, text=get_sample(file)))

    @property
    def docs(self) -> list[Document]:
        return [*self._docs] # shallow copy

    @property
    def titles(self) -> list[str]:
        return [d.title for d in self._docs]

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self._docs]

    @property
    def vectors(self) -> list[SparseVector]:
        return [d.vector for d in self._docs]

    def get(self, index: int) -> Document:
        return self._docs[index]

    def find(self, title: str) -> int:
        return ([d.title for d in self._docs]).index(title)

    def select(self, indices: int) -> dict[int, Document]:
        result = {}
        for i in indices:
            result[i] = self.get(i)
        return result

    def add_vector(self, index: int, vector: SparseVector) -> None:
        self.get(index).vector = vector
