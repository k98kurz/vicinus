# vicinus

## Classes

### `SparseVector`

Class representing a sparse vector, i.e. omitting every element that equals 0.
Useful for high dimensionality vectors with mostly 0s.

#### Methods

##### `__init__(data: dict[int, int | float] = None):`

Initialize. Raises `TypeError` for invalid data.

##### `get(index: int = 0) -> int | float | None:`

Get the element at the specified index.

##### `keys() -> set[int]:`

Return the set of indices at which elements exist.

##### `values() -> ValuesView[int | float]:`

Return a view of all elements.

##### `items() -> ItemsView[tuple[int, int | float]]:`

Return an ItemsView for `for i, e in vec.items()` syntax.

##### `norm() -> float:`

Calculate the Euclidean norm of a SparseVector, defined as the square root of
the dot product of itself (i.e. the sqrt of the sum of the squares of the
elements).

##### `dot_product(other: SparseVector) -> float:`

Calculate the dot product between two `SparseVector`s. Raises `TypeError` for
invalid other. Treats unset indices as 0 values mathematically.

##### `cosine_similarity(other: SparseVector) -> float:`

Calculate the cosine similarity with another SparseVector, defined as the doc
product divided by the product of norms. Returns 0 if the product of norms is
zero. Raises `TypeError` for invalid `other`.

##### `copy() -> SparseVector:`

Return a copy of the `SparseVector`.

### `VectorDB`

Class implementing an in-memory vector database. Can be configured for Jaccard,
NGF, or NGF-IDF (NGF by default), N-Gram size (default is 5), and IDF mode
(either saving space or saving N-Gram counts to save on index computational
costs of CRUD ops; `IDFMode.SAVE_SPACE` is default).

#### Annotations

- N: <class 'int'>
- mode: <enum 'VDBMode'>
- idf_mode: <enum 'IDFMode'>
- corpus: dict[typing.Hashable, str]
- vectors: dict[typing.Hashable, vicinus.sparse_vector.SparseVector | set[str]]
- idf: vicinus.sparse_vector.SparseVector | None
- ng_counts: dict[typing.Hashable, vicinus.sparse_vector.SparseVector] | None

#### Methods

##### `__init__(N: int = 5, /, *, idf_mode: IDFMode = 0, mode: VDBMode = 0):`

Initialize with the given configuration. Raises `TypeError` or `ValueError` for
invalid arguments.

##### `recalculate(replace: bool = False):`

Recalculate the "index", i.e. the `SparseVector`s used for cosine similarity for
`VDBMode.NGF`/`VDBMode.NGF_IDF` or `set[str]`s for `VDBMode.JACCARD`. By
default, only fills in the missing vectors/sets; pass `replace=True` to
recalculate all vectors/sets.

##### `set_corpus(corpus: dict, vectors: dict = None, idf: SparseVector = None, ng_counts: dict = None, recalculate: bool = True):`

Set the initial corpus. `corpus` must be a dict mapping titles/ids to contents.
To avoid recalculations when restoring db state from persistent storage, pass
`vectors` (for any `VDBMode`), `idf` (for `VDBMode.NGF_IDF`), and/or `ng_counts`
(for `VDBMode.NGF_IDF`). `vectors` is an optional dict mapping the titles/ids to
`SparseVector`s or `set[str]` (for `VDBMode.JACCARD`). `idf` is the IDF
`SparseVector`. `ng_counts` is a dict mapping titles/ids to `SparseVector`
N-Gram counts. If `recalculate=True` (default), calls `self.recalculate()`.
Raises `TypeError` or `ValueError` for invalid arguments.

##### `add(vector_id: Hashable, content: str, recalculate: bool = True):`

Add a record to the vector db. If recalculate is `True`, `self.recalculate()`
will be called (default behavior).

##### `get(vector_id: Hashable) -> tuple[str, list[vicinus.sparse_vector.SparseVector | set]]:`

Get a specific record by its id. Returns a tuple of the form `(content, [vector])`
or `(content, [vector, ng_count])` (vector will bet `set[str]` for
`VDBMode.JACCARD`). Raises `IndexError` if the id is not found.

##### `update(vector_id: Hashable, content: str, recalculate: bool = True):`

Update a record, replacing its content. If the record does not yet exist, add
it. If `recalculate=True` (default), then `self.recalculate()` will be called.

##### `remove(vector_id: Hashable, recalculate: bool = True):`

Remove a record by its id. If it does not exist, no-op. If `recalculate=True`
(default), then `self.recalculate()` will be called.

##### `search(query: str, limit: int = 4) -> list[tuple[float, typing.Hashable]]:`

Search through the records using the appropriate algorithm. Results are ordered
by descending similarity score. Returns a list of tuples of form `(score, id)`.
Set `limit=-1` to return all results. Raises `TypeError` for non-int `limit`.

### `VDBMode(IntEnum)`

Enum representing valid VectorDB modes: NGF, NGF_IDF, or JACCARD.

### `IDFMode(IntEnum)`

Enum representing valid modes for NGF-IDF calculation: either SAVE_SPACE for
default behavior or SAVE_COUNTS for reusing the N-Gram counts during
recalculation.

## Functions

### `rank(score_fn: Callable, candidates: dict) -> list[tuple[float, typing.Any]]:`

Ranks `candidates` using the `score_fn` callable. Returns a list of tuples in
form `(score: float, index: int)`, sorted by score descending.

### `select(rankings: list, k: int = 4) -> list[tuple[float, int]]:`

Select the top `k` from the rankings. Sorts beforehand. Assumes similarity
scores and not distance scores. Raises `TypeError` or `ValueError` for invalid
inputs.

### `tokenize(text: str) -> list[str]:`

Preprocess some text by casting to lower, removing punctuation, and splitting on
word boundaries. Returns only alphanumeric tokens.

### `hamming_distance(str1: str, str2: str, normalize: bool = False) -> int | float:`

Calculate Hamming distance between two strs. If `normalize` is `False`
(default), distance is returned directly as an int. If `normalize` is set to
`True`, the distance is normalized to a float proportional to the length of the
longest string.

### `hamming_similarity(str1: str, str2: str) -> float:`

Hamming similarity: 1 - normalized Hamming distance.

### `levenshtein_distance(a: str, b: str, normalize: bool = False) -> int | float:`

Calculate Levenshtein distance between two strs. If `normalize` is `False`
(default), distance is returned directly as an int. If `normalize` is set to
`True`, the distance is normalized to a float proportional to the average length
of the strings, clamped to a max of 1.0. Optimized with dynamic programming.

### `levenshtein_similarity(a: str, b: str) -> float:`

Turns the normalized Levenshtein difference metric into a similarity metric by
substracting it from 1.0.

### `ngf_index(ng: str) -> int:`

Calculate the NGF vector index for a given alphanumeric N-Gram.

### `ng_count(text: str, N: int = 3, ngs: set = None) -> SparseVector:`

Calculate the N-Gram Count of some text. First step in NGF or NGF-IDF analysis,
where NGF is like TF but with N-Grams instead of whole words. The `N` parameter
controls N-Gram size. Produces a SparseVector with max length `36^N`, where
element index is ordered by N-Gram alphanumeric order, e.g. 000, 001, ..., 0a0,
0a1, ..., zzz.

### `ngf(text: str, N: int = 3, vec: SparseVector = None) -> SparseVector:`

Calculate the N-Gram Frequency of some text. Analogous to TF, but with N-Grams
instead of whole words. The `N` parameter controls N-Gram size. Produces a
SparseVector max length `36^N`, where element index is ordered by N-Gram
alphanumeric order, e.g. 000, 001, ..., 0a0, 0a1, ..., zzz. Pass `vec` result of
`ng_count` to bypass retokenization of the text and recalculation of N-Grams.

### `ngf_rank(query: str, text_vecs: list, N: int = 3) -> list[tuple[float, int]]:`

Runs NGF cosine similarity between the query and the text vectors. Returns a
sorted list of form `[(similarity, index)]`.

### `ngf_select(query: str, text_vecs: dict, k: int = 4, N: int = 3) -> list[tuple[float, int]]:`

Runs NGF cosine similarity between the query and the text vectors. Returns the
top `k` candidates in a sorted list of form `[(similarity, key)]`.

### `ngf_idf_setup(corpus: dict, N: int = 3, ng_counts: dict = None) -> tuple[vicinus.sparse_vector.SparseVector, dict[typing.Any, vicinus.sparse_vector.SparseVector], dict[typing.Any, vicinus.sparse_vector.SparseVector]]:`

Processes a corpus, creating the vectors for each text in a pipeline: 1) extract
N-Grams and create NG count vectors; 2) aggregate corpus NG count vector; 3)
create NGF vectors; 4) calculate the IDF vector; 5) modify NGF vectors by
multiplying by the IDF vector. Returns the corpus IDF vector as well as the
NGF-IDF vectors to enable fuzzy search on queries.

### `ngf_idf_query(query: str, corpus_idf: SparseVector, N: int = 3) -> SparseVector:`

Processes a query, calculating the NGF-IDF vector for it using the corpus IDF
vector.

### `ngf_idf_rank(query: str, corpus_idf: SparseVector, text_vecs: dict, N: int = 3) -> list[tuple[float, typing.Any]]:`

Runs NGF-IDF cosine similarity between the query and the text vectors. Returns a
sorted list of form `[(similarity, key)]`.

### `ngf_idf_select(query: str, corpus_idf: SparseVector, text_vecs: dict, k: int = 4, N: int = 3) -> list[tuple[float, int]]:`

Runs NGF-IDF cosine similarity between the query and the text vectors. Returns
the top `k` candidates in a sorted list of form `[(similarity, index)]`.

### `n_grams(text: str, N: int = 3) -> set[str]:`

Extract the N-Grams of a text, defined as the set of unique substrings of the
given N.

### `jaccard_index(a: set, b: set) -> float:`

Calculate the Jaccard similarity index between two sets, e.g. of N-Grams.
Defined as the cardinality of the intersection divided by the cardinality of the
union.

### `list_samples() -> list[str]:`

Returns the list of bundled sample file names.

### `get_sample(name: str) -> str:`

Reads and returns the content of the named sample. Raises FileNotFoundError for
an invalid name.

### `version() -> str:`

Return the current library version.


