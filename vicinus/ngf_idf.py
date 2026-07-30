from math import log
from typing import Any
from vicinus.errors import type_assert, value_assert
from vicinus.core import rank, select, tokenize
from vicinus.ngrams import n_grams
from vicinus.sparse_vector import SparseVector


def ngf_index(ng: str) -> int:
    """Calculate the NGF vector index for a given alphanumeric N-Gram."""
    N = len(ng)
    ind = 0
    for i in range(N):
        c = ng[i]
        if '0' <= c <= '9':
            ind += (ord(c) - ord('0')) * (10 ** (N-i-1))
        elif 'a' <= c <= 'z':
            ind += (10 + ord(c) - ord('a')) * (10 ** (N-i-1))
    return ind


def ng_count(text: str, N: int = 3, ngs: set[str] = None) -> SparseVector:
    """Calculate the N-Gram Count of some text. First step in NGF or
        NGF-IDF analysis, where NGF is like TF but with N-Grams instead
        of whole words. The `N` parameter controls N-Gram size.
        Produces a SparseVector with max length `36^N`, where element
        index is ordered by N-Gram alphanumeric order, e.g. 000, 001,
        ..., 0a0, 0a1, ..., zzz.
    """
    text = ''.join(tokenize(text))
    ngs = ngs or n_grams(text, N)
    type_assert(type(ngs) is set, 'ngs must be set[str]')
    vec = SparseVector()
    for ng in ngs:
        vec[ngf_index(ng)] = text.count(ng)
    return vec


def ngf(text: str, N: int = 3, vec: SparseVector = None) -> SparseVector:
    """Calculate the N-Gram Frequency of some text. Analogous to TF, but
        with N-Grams instead of whole words. The `N` parameter
        controls N-Gram size. Produces a SparseVector max length
        `36^N`, where element index is ordered by N-Gram alphanumeric
        order, e.g. 000, 001, ..., 0a0, 0a1, ..., zzz. Pass `vec` result
        of `ng_count` to bypass retokenization of the text and
        recalculation of N-Grams.
    """
    vec = vec or ng_count(text, N=N)
    type_assert(isinstance(vec, SparseVector), 'vec must be SparseVector')
    total = sum(vec.values())
    new_vec = SparseVector()
    for i in vec:
        new_vec[i] = vec[i] / total
    return new_vec


def ngf_rank(
        query: str, text_vecs: list[SparseVector], N: int = 3
    ) -> list[tuple[float, int]]:
    """Runs NGF cosine similarity between the query and the text
        vectors. Returns a sorted list of form `[(similarity, index)]`.
    """
    qvec = ngf(query, N=N)
    return rank(qvec.cosine_similarity, text_vecs)


def ngf_select(
        query: str, text_vecs: list[SparseVector], k: int = 4, N: int = 3
    ) -> list[tuple[float, int]]:
    """Runs NGF cosine similarity between the query and the text
        vectors. Returns the top `k` candidates in a sorted list of form
        `[(similarity, index)]`.
    """
    ranks = ngf_rank(query, text_vecs, N)
    return select(ranks, k)


def ngf_idf_setup(
        corpus: dict[Any, str], N: int = 3,
        ng_counts: dict[Any, SparseVector] = None,
    ) -> tuple[SparseVector, dict[Any, SparseVector], dict[Any, SparseVector]]:
    """Processes a corpus, creating the vectors for each text in a
        pipeline: 1) extract N-Grams and create NG count vectors; 2)
        aggregate corpus NG count vector; 3) create NGF vectors; 4)
        calculate the IDF vector; 5) modify NGF vectors by multiplying
        by the IDF vector. Returns the corpus IDF vector as well as the
        NGF-IDF vectors to enable fuzzy search on queries.
    """
    type_assert(type(corpus) is dict, 'corpus must be dict[Any, str]')
    type_assert(all([type(v) is str for v in corpus.values()]),
        'corpus must be dict[Any, str]')
    type_assert(type(N) is int, 'N must be int')
    value_assert(N > 0, 'N must be >0')

    # 1: NG counts per text
    ng_counts = ng_counts or {}
    ngcs = {
        k: ng_counts[k] if k in ng_counts else ng_count(text, N=N)
        for k, text in corpus.items()
    }

    # 2: aggregate corpus NG count
    indices = set()
    for ngc in ngcs.values():
        indices = indices | ngc.keys()
    corpus_counts = SparseVector({
        i: sum([ngc.get(i, 0) for ngc in ngcs.values()])
        for i in indices
    })

    # 3: NGF vectors
    vecs = {
        k: ngf(v, N=N, vec=ngcs[k])
        for k, v in corpus.items()
    }

    # 4: IDF
    N = len(corpus) + 1
    idf = SparseVector()
    for i in corpus_counts:
        n = sum([1 if i in v else 0 for v in vecs.values()]) + 1
        idf[i] = log(N / n)

    # 5: NGF-IDF
    for k, vec in vecs.items():
        ngc = ngcs[k]
        for k in ngc:
            vec[k] *= idf[k]

    return (idf, vecs, ngcs)


def ngf_idf_query(
        query: str, corpus_idf: SparseVector, N: int = 3
    ) -> SparseVector:
    """Processes a query, calculating the NGF-IDF vector for it using
        the corpus IDF vector.
    """
    # 1: NG count for query
    ngc = ng_count(query, N=N)

    # 2: NGF vector
    vec = ngf(query, N=N, vec=ngc)

    # 3: NGF-IDF
    for k, v in ngc.items():
        vec[k] *= corpus_idf.get(k, 0)

    return vec


def ngf_idf_rank(
        query: str, corpus_idf: SparseVector, text_vecs: dict[Any, SparseVector],
        N: int = 3
    ) -> list[tuple[float, Any]]:
    """Runs NGF-IDF cosine similarity between the query and the text
        vectors. Returns a sorted list of form `[(similarity, key)]`.
    """
    qvec = ngf_idf_query(query, corpus_idf, N)
    return rank(qvec.cosine_similarity, text_vecs)


def ngf_idf_select(
        query: str, corpus_idf: SparseVector, text_vecs: dict[Any, SparseVector],
        k: int = 4, N: int = 3
    ) -> list[tuple[float, int]]:
    """Runs NGF-IDF cosine similarity between the query and the text
        vectors. Returns the top `k` candidates in a sorted list of form
        `[(similarity, index)]`.
    """
    ranks = ngf_idf_rank(query, corpus_idf, text_vecs, N)
    return select(ranks, k)
