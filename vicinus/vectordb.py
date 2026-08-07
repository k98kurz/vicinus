from enum import IntEnum
from typing import Hashable
from vicinus.core import rank, select
from vicinus.ctf_idf import ctf, ctf_rank, ctf_idf_setup, ctf_idf_rank
from vicinus.errors import type_assert, value_assert
from vicinus.ngf_idf import ngf, ngf_rank, ngf_idf_setup, ngf_idf_rank
from vicinus.ngrams import n_grams, jaccard_index
from vicinus.sparse_vector import SparseVector


class VDBMode(IntEnum):
    """Enum representing valid VectorDB modes: NGF, NGF_IDF, or JACCARD."""
    NGF = 0
    NGF_IDF = 1
    JACCARD = 2
    CTF = 3
    CTF_IDF = 4


class IDFMode(IntEnum):
    """Enum representing valid modes for NGF-IDF/CTF-IDF calculation: either
        SAVE_SPACE for default behavior or SAVE_COUNTS for reusing the
        N-Gram/CRC(Token) counts during recalculation.
    """
    SAVE_SPACE = 0
    SAVE_COUNTS = 1


class VectorDB:
    """Class implementing an in-memory vector database. Can be
        configured for Jaccard, NGF, NGF-IDF, CTF, or CTF-IDF (NGF by
        default), N-Gram size/CT dimensions (default is 5 or 10000,
        respectively), and IDF mode (either saving space or saving
        N-Gram counts to save on index computational costs of CRUD ops;
        `IDFMode.SAVE_SPACE` is default).
    """
    N: int|None
    mode: VDBMode
    idf_mode: IDFMode
    corpus: dict[Hashable, str]
    vectors: dict[Hashable, SparseVector|set[str]]
    idf: SparseVector | None
    counts: dict[Hashable, SparseVector] | None

    def __init__(
            self, N: int = None, *,
            mode: VDBMode = VDBMode.NGF, idf_mode: IDFMode = IDFMode.SAVE_SPACE,
        ):
        """Initialize with the given configuration. Raises `TypeError`
            or `ValueError` for invalid arguments. For `VDBMode.CTF` or
            `VDBMode.CTF_IDF`, pass `N=-1` to use `2**32` dimensions
            (full crc32 range) instead of reducing dimensionality by
            modulo.
        """
        type_assert(isinstance(mode, VDBMode), 'mode must be VDBMode')
        self.mode = mode

        type_assert(isinstance(idf_mode, IDFMode), 'idf_mode must be IDFMode')
        self.idf_mode = idf_mode

        type_assert(type(N) is int or N is None, 'N must be int|None')
        if self.mode in (VDBMode.CTF, VDBMode.CTF_IDF):
            if N is None:
                N = 10_000
            elif N < 0:
                N = None
            else:
                value_assert(N > 1, 'N must be -1, None, or >1')
        elif N is None: # default for NGF, NGF_IDF, JACCARD
            N = 5
        if self.mode not in (VDBMode.CTF, VDBMode.CTF_IDF):
            value_assert(N > 1, f'N must be >1 for {self.mode.name}')
        self.N = N

        self.corpus = {}
        self.vectors = {}
        self.idf = None
        self.counts = None

        if self.mode in (VDBMode.NGF_IDF, VDBMode.CTF_IDF):
            self.idf = SparseVector()
            self.counts = {}

    def recalculate(self, replace: bool = False):
        """Recalculate the "index", i.e. the `SparseVector`s used for
            cosine similarity for `VDBMode.NGF`/`VDBMode.NGF_IDF` or
            `set[str]`s for `VDBMode.JACCARD`. By default, only fills in
            the missing vectors/sets; pass `replace=True` to recalculate
            all vectors/sets.
        """
        type_assert(type(replace) is bool, 'replace must be bool')

        if self.mode is VDBMode.JACCARD:
            for k, v in self.corpus.items():
                if k not in self.vectors or replace:
                    self.vectors[k] = n_grams(v, self.N)
            return

        if self.mode is VDBMode.NGF:
            for k, v in self.corpus.items():
                if k not in self.vectors or replace:
                    self.vectors[k] = ngf(v, self.N)
            return

        if self.mode is VDBMode.NGF_IDF:
            if len(self.corpus) == len(self.vectors) and self.idf and not replace:
                return # already set
    
            if replace:
                idf, vectors, counts = ngf_idf_setup(self.corpus, self.N)
            else:
                idf, vectors, counts = ngf_idf_setup(
                    self.corpus, self.N, self.counts
                )
            self.idf = idf
            self.vectors = vectors
            if self.idf_mode is IDFMode.SAVE_COUNTS:
                self.counts = counts
            return

        if self.mode is VDBMode.CTF:
            for k, v in self.corpus.items():
                if k not in self.vectors or replace:
                    self.vectors[k] = ctf(v, self.N)
            return

        if self.mode is VDBMode.CTF_IDF:
            if len(self.corpus) == len(self.vectors) and self.idf and not replace:
                return # already set
    
            if replace:
                idf, vectors, counts = ctf_idf_setup(self.corpus, self.N)
            else:
                idf, vectors, counts = ctf_idf_setup(
                    self.corpus, self.N, self.counts
                )
            self.idf = idf
            self.vectors = vectors
            if self.idf_mode is IDFMode.SAVE_COUNTS:
                self.counts = counts
            return

    def set_corpus(
            self, corpus: dict[Hashable, str],
            vectors: dict[Hashable, SparseVector|set[str]] = None,
            idf: SparseVector = None,
            counts: dict[Hashable, SparseVector] = None,
            recalculate: bool = True,
        ) -> None:
        """Set the initial corpus. `corpus` must be a dict mapping
            titles/ids to contents. To avoid recalculations when
            restoring db state from persistent storage, pass `vectors`
            (for any `VDBMode`), `idf` (for `VDBMode.NGF_IDF`), and/or
            `counts` (for `VDBMode.NGF_IDF`). `vectors` is an
            optional dict mapping the titles/ids to `SparseVector`s or
            `set[str]` (for `VDBMode.JACCARD`). `idf` is the IDF
            `SparseVector`. `counts` is a dict mapping titles/ids
            to `SparseVector` N-Gram counts. If `recalculate=True`
            (default), calls `self.recalculate()`. Raises `TypeError` or
            `ValueError` for invalid arguments.
        """
        type_assert(isinstance(corpus, dict), 'corpus must be dict[str, str]')
        type_assert(all([type(v) is str for v in corpus.values()]),
            'corpus must be dict[str, str]')

        if vectors is not None:
            type_assert(isinstance(vectors, dict),
                'vectors must be dict[Hashable, SparseVector]')
            type_assert(
                all([
                    type(v) in (SparseVector, set) for v in vectors.values()
                ]),
                'vectors must be dict[Hashable, SparseVector]'
            )
            value_assert(all([k in corpus for k in vectors]),
                '`vectors` keys must match `corpus` keys')

        if idf is not None:
            type_assert(isinstance(idf, SparseVector),
                'idf must be SparseVector')

        if counts is not None:
            type_assert(isinstance(counts, dict),
                'counts must be dict[str, SparseVector]')
            type_assert(all([
                    isinstance(v, SparseVector) for v in counts.values()
                ]),
                'counts must be dict[Hashable, SparseVector]'
            )
            value_assert(all([k in corpus for k in counts]),
                '`counts` keys must match `corpus` keys')

        self.corpus = {**corpus}
        self.vectors = {**vectors} if vectors else {}
        self.idf = idf if idf else None
        self.counts = counts if counts else {}

        return self.recalculate() if recalculate else None

    def add(
            self, vector_id: Hashable, content: str, recalculate: bool = True
        ) -> None:
        """Add a record to the vector db. If recalculate is `True`,
            `self.recalculate()` will be called (default behavior).
        """
        if vector_id in self.corpus:
            return

        self.corpus[vector_id] = content
        return self.recalculate() if recalculate else None

    def get(self, vector_id: Hashable) -> tuple[str, list[SparseVector|set]]:
        """Get a specific record by its id. Returns a tuple of the form
            `(content, [vector])` or `(content, [vector, ng_count])`
            (vector will bet `set[str]` for `VDBMode.JACCARD`). Raises
            `IndexError` if the id is not found.
        """
        if vector_id not in self.corpus:
            raise IndexError(f"vector_id={vector_id} not found")

        vecs = []
        if vector_id in self.vectors:
            vecs.append(self.vectors[vector_id])
        if self.counts and vector_id in self.counts:
            vecs.append(self.counts[vector_id])

        return self.corpus[vector_id], vecs

    def update(
            self, vector_id: Hashable, content: str, recalculate: bool = True
        ) -> None:
        """Update a record, replacing its content. If the record does
            not yet exist, add it. If `recalculate=True` (default), then
            `self.recalculate()` will be called.
        """
        if vector_id not in self.corpus:
            return self.add(vector_id, content, recalculate)

        if self.corpus[vector_id] == content:
            return

        self.corpus[vector_id] = content

        if vector_id in self.vectors:
            del self.vectors[vector_id]

        if self.counts and vector_id in self.counts:
            del self.counts[vector_id]

        return self.recalculate() if recalculate else None

    def remove(self, vector_id: Hashable, recalculate: bool = True) -> None:
        """Remove a record by its id. If it does not exist, no-op. If
            `recalculate=True` (default), then `self.recalculate()` will
            be called.
        """
        if vector_id not in self.corpus:
            return

        del self.corpus[vector_id]

        if vector_id in self.vectors:
            del self.vectors[vector_id]

        if self.counts and vector_id in self.counts:
            del self.counts[vector_id]

        if recalculate and self.mode in (VDBMode.NGF_IDF, VDBMode.CTF_IDF):
            self.idf = None

        return self.recalculate() if recalculate else None

    def search(self, query: str, limit: int = 4) -> list[tuple[float, Hashable]]:
        """Search through the records using the appropriate algorithm.
            Results are ordered by descending similarity score. Returns
            a list of tuples of form `(score, id)`. Set `limit=-1` to
            return all results. Raises `TypeError` for non-int `limit`.
        """
        if self.mode is VDBMode.JACCARD:
            q = n_grams(query, self.N)
            rankings = rank(lambda s: jaccard_index(q, s), self.vectors)
        elif self.mode is VDBMode.NGF:
            rankings = ngf_rank(query, self.vectors, self.N)
        elif self.mode is VDBMode.NGF_IDF:
            rankings = ngf_idf_rank(query, self.idf, self.vectors, self.N)
        elif self.mode is VDBMode.CTF:
            rankings = ctf_rank(query, self.vectors, self.N)
        elif self.mode is VDBMode.CTF_IDF:
            rankings = ctf_idf_rank(query, self.idf, self.vectors, self.N)

        return rankings if limit < 1 else select(rankings, limit)
