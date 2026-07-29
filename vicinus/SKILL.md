---
name: vicinus
description: Guide for using the Vicinus library for fuzzy search.
metadata:
  version: 0.0.1
  last-updated: 2026-07-28
  author: "Jonathan Voss"
  library-name: "vicinus"
  repository: "https://github.com/k98kurz/vicinus"
---

# Vicinus

## Hamming Distance/Similarity

The simplest and least costly algorithm for comparing two texts is Hamming Distance,
which can be normalized into a similarity metric. It is the least robust.

```python
from vicinus import hamming_distance, hamming_similarity

distance = hamming_distance(text1, text2)
similarity = hamming_similarity(text1, text)
```

## Levenshtein Distance/Similarity

Levenshtein Distance is more robust than Hamming, but it is more computationally
costly (recursive) Performance was improved by using memoization on the function,
but the `lru_cache` was limited in size to prevent memory explosion. Should not be
used for large texts.

```python
from vicinus import levenshtein_distance, levenshtein_similarity

distance = levenshtein_distance(text1, text2)
similarity = levenshtein_similarity(text1, text)
```

## N-Grams + Jaccard Index

Another option which scales better than Levenshtein for large texts. N-Grams should
be at least 3 long (trigrams) but can be made longer for larger texts. Jaccard Index
is a measure of similarity based upon set inclusion of N-Grams: shared/total
(|intersection| / |union|). Only aphanumeric chars (standard Latin lowercase +
Arabic numerals) are supported.

```python
from vicinus import n_grams, jaccard_index

ng1 = n_grams(text1, N=3) # default N
ng2 = n_grams(text2, N=3)
similarity = jaccard_index(ng1, ng2) # between 0 and 1.0
```

## NGF (Cosine Similarity)

Similar to TF, but with N-Grams instead of Terms. Uses `SparseVector`s for
efficiency. Uses `n_grams` but not `jaccard_index`. Does not involve a setup phase
that must be rerun when a text in the corpuse changes; only that text must be
reprocessed.

```python
from vicinus import ngf, ngf_rank, ngf_select

corpus = ["Text 1 is some text...", "Text 2 is another thing...", ...]
vecs = [ngf(t) for t in corpus]
query = "text about something"
# to sort all texts
rankings = ngf_rank(query, vecs)
# or select top k=2
candidates = ngf_select(query, vecs, k=2) # or candidates = select(rankings, 2)
```

## NGF-IDF (Cosine Similarity)

Similar to TF-IDF, but with N-Grams instead of Terms. Uses `SparseVector`s for
efficiency. Uses `n_grams` but not `jaccard_index`. Involves a setup phase in which
the corpus is processed to derive a corpus IDF vector and NGF-IDF vectors, then a
query phase once the corpus IDF vector and individual text NGF-IDF vectors are
available. Ranking is done via cosine similarity. Adding or editing a text in the
corpus requires rerunning setup on the whole corpus.

```python
from vicinus import ngf_idf_setup, ngf_idf_rank, ngf_idf_select

corpus = ["Text 1 is some text...", "Text 2 is another thing...", ...]
corpus_idf, vecs = ngf_idf_setup(corpus, N=3) # default N
query = "text about something"
# to sort all texts by cosine similarity
all_ranked = ngf_idf_rank(query, corpus_idf, vecs, N=3) # default N
# or select the top k=2
candidates = ngf_idf_select(query, corpus_idf, vecs, 2, N=3)
```

Return values from `ngf_idf_rank` and `ngf_idf_select` have form
`[(score, index),]`, where `index` is the index of the text vector with the given
score. `ngf_idf_rank` returns rankings sorted in descending score order.

## Helpers: `rank` and `select`

Helpers for ranking candidates and selecting the top k of rankings.

```python
from vicinus import rank, select, hamming_similarity

texts = [...]
query = "something"
rankings = rank(lambda t: hamming_similarity(query, t), texts)
candidates = select(rankings, 2)
```

## Advanced Use

The library exposes a `SparseVector` class which has the following methods:
- `__init__(self, data: dict[int, int|float] = None)`
- `__getitem__(self, index: int) -> int|float|None`
- `__setitem__(self, index: int, value: int|float) -> None`
- `__delitem__(self, index: int) -> None`
- `__iter__(self) -> Iterator`
- `__len__(self) -> int`
- `__repr__(self) -> str`
- `get(self, index: int, default = None) -> int|float|None`
- `keys(self) -> set[int]`
- `values(self) -> ValuesView[int|float]`
- `items(self) -> ItemsView[tuple[int, int|float]]`
- `norm(self) -> float`
- `dot_product(self, other: SparseVector) -> float`
- `cosine_similarity(self, other: SparseVector) -> float`

The library exposes some additional functions that may be useful for experimentation.
- `tokenize(text: str) -> list[str]`: split, lowercase, strip non-alnum
- `ng_count(text: str, N: int=3, ngs: str[str]=None) -> SparseVector`
- `ngf_index(ng: str) -> int`: calculate the NGF vec index for a given alnum N-Gram
- `ngf_idf_query(query: str, corpus_idf: SparseVector, N: int=3) -> SparseVector`:
  proceses a query into an NGF-IDF vector; used by `ngf_idf_rank` and `ngf_idf_select`
