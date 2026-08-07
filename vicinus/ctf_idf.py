from math import log
from typing import Any
from vicinus.core import rank, select, tokenize
from vicinus.errors import type_assert, value_assert
from vicinus.sparse_vector import SparseVector
from zlib import crc32


def ctf_index(token: str, N: int = None) -> int:
    """Calculate the CRC vector index for a given token. With `N=None`,
        dimensionality will be `2**32` (the full range of crc32).
    """
    return crc32(token.encode()) % N if N else crc32(token.encode())


def ct_count(text: str, N: int|None = 10000) -> SparseVector:
    """Calculate the CRC(Token) Count of some text. Set `N=None` to use
        `2**32` dimensions.
    """
    type_assert(type(text) is str, 'text must be str')
    type_assert(type(N) is int or N is None, 'N must be int|None')
    value_assert(N > 1, 'N must be >1') if N is not None else N
    indices = [ctf_index(t, N) for t in tokenize(text)]
    vec = SparseVector()
    for i in indices:
        vec[i] += 1
    return vec


def ctf(
        text: str, N: int|None = 10000, ctc: SparseVector = None
    ) -> SparseVector:
    """Calculate the CRC(Token) Frequency of some text. Effectively a
        version of TF with `N` upper limit on dimensionality. Set
        `N=None` to use `2**32` dimensions.
    """
    type_assert(type(text) is str, 'text must be str')
    type_assert(type(N) is int or N is None, 'N must be int|None')
    value_assert(N > 1, 'N must be >1') if N is not None else N
    ctc = ctc or ct_count(text, N)
    total = sum(ctc.values())
    return SparseVector({
        i: c / total
        for i, c in ctc.items()
    })


def ctf_rank(
        query: str, text_vecs: dict[Any, SparseVector], N: int|None = 10000
    ) -> list[tuple[float, int]]:
    """Runs CTF cosine similarity between the query and the text
        vectors. Returns a sorted list of form `[(similarity, key)]`,
        where `key` is a key of `text_vecs`. Set `N=None` to use `2**32`
        dimensions.
    """
    qvec = ctf(query, N=N)
    return rank(qvec.cosine_similarity, text_vecs)


def ctf_select(
        query: str, text_vecs: dict[Any, SparseVector], k: int = 4, N: int|None = 10000
    ) -> list[tuple[float, int]]:
    """Runs CTF cosine similarity between the query and the text
        vectors. Returns the top `k` candidates in a sorted list of form
        `[(similarity, key)]`, where `key` is a key of `text_vecs`. Set
        `N=None` to use `2**32` dimensions.
    """
    ranks = ctf_rank(query, text_vecs, N=N)
    return select(ranks, k)


def ctf_idf_setup(
        corpus: dict[Any, str], N: int|None = 10000,
        ct_counts: dict[Any, SparseVector] = None,
    ) -> tuple[SparseVector, dict[Any, SparseVector], dict[Any, SparseVector]]:
    """Processes a corpus, creating the vectors for each text in a
        pipeline: 1) create the CRC(Token) count vector for each text;
        2) aggregate corpus CT count vector; 3) create CTF vector for
        each text; 4) calculate the IDF vector; 5) modify CTF vectors by
        multiplying by the IDF vector. Returns the corpus IDF vector as
        well as the CTF-IDF and CT count vectors for each text. Use the
        IDF and CTF-IDF vectors to do fuzzy search; save and re-use
        CT count vectors to skip step 1 during updates. Set `N=None` to
        use `2**32` dimensions.
    """
    type_assert(type(corpus) is dict, 'corpus must be dict[Any, str]')
    type_assert(all([type(v) is str for v in corpus.values()]),
        'corpus must be dict[Any, str]')
    type_assert(type(N) is int or N is None, 'N must be int|None')
    value_assert(N > 1, 'N must be >1') if N is not None else N

    # 1: CRC-Token counts per text
    ct_counts = ct_counts or {}
    ctcs = {
        k: ct_counts[k] if k in ct_counts else ct_count(text, N=N)
        for k, text in corpus.items()
    }

    # 2: aggregate corpus CT count
    indices = set()
    for ctc in ctcs.values():
        indices = indices | ctc.keys()
    corpus_counts = SparseVector({
        i: sum([ctc[i] for ctc in ctcs.values()])
        for i in indices
    })

    # 3: CTF vectors
    vecs = {
        k: ctf(v, N=N, ctc=ctcs[k])
        for k, v in corpus.items()
    }

    # 4: IDF
    N = len(corpus) + 1
    idf = SparseVector()
    for i in corpus_counts:
        n = sum([1 if i in v else 0 for v in vecs.values()]) + 1
        idf[i] = log(N / n)

    # 5: CTF-IDF
    for k, vec in vecs.items():
        ctc = ctcs[k]
        for k in ctc:
            vec[k] *= idf[k]

    return (idf, vecs, ctcs)


def ctf_idf_query(
        query: str, corpus_idf: SparseVector, N: int|None = 10000
    ) -> SparseVector:
    """Processes a query, calculating the CTF-IDF vector for it using
        the corpus IDF vector. Set `N=None` to use `2**32` dimensions.
    """
    type_assert(type(query) is str, 'query must be str')
    type_assert(isinstance(corpus_idf, SparseVector),
        'corpus_idf must be SparseVector')
    type_assert(type(N) is int or N is None, 'N must be int|None')
    value_assert(N > 1, 'N must be >1') if N is not None else N
    # 1: CT count for query
    ctc = ct_count(query, N=N)

    # 2: CTF vector
    vec = ctf(query, N=N, ctc=ctc)

    # 3: CTF-IDF
    for k, v in ctc.items():
        vec[k] *= corpus_idf.get(k, 0)

    return vec


def ctf_idf_rank(
        query: str, corpus_idf: SparseVector, text_vecs: dict[Any, SparseVector],
        N: int|None = 10000
    ) -> list[tuple[float, Any]]:
    """Runs CTF-IDF cosine similarity between the query and the text
        vectors. Returns a sorted list of form `[(similarity, key)]`,
        where `key` is a key of `text_vecs`. Set `N=None` to use `2**32`
        dimensions.
    """
    qvec = ctf_idf_query(query, corpus_idf, N)
    return rank(qvec.cosine_similarity, text_vecs)


def ctf_idf_select(
        query: str, corpus_idf: SparseVector, text_vecs: dict[Any, SparseVector],
        k: int = 4, N: int|None = 10000
    ) -> list[tuple[float, int]]:
    """Runs CTF-IDF cosine similarity between the query and the text
        vectors. Returns the top `k` candidates in a sorted list of form
        `[(similarity, key)]`, where `key` is a key of `text_vecs`. Set
        `N=None` to use `2**32` dimensions.
    """
    ranks = ctf_idf_rank(query, corpus_idf, text_vecs, N)
    return select(ranks, k)
