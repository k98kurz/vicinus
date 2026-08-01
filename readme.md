# Vicinus

Vicinus is a library for building fuzzy search tools. It includes multiple
similarity/distance algorithms, an in-memory vector DB, and an optional
persistent vector DB using my sqloquent ORM package (sqlite3). Made from
all-organic, artisanal, 100% hand-crafted code with no gen AI (except to write
some text samples for testing) and no external dependencies.

This library includes several standard algorithms as well as a few novel (to my
knowledge) constructions: NGF and NGF-IDF, adapted from TF and TF-IDF but using
N-Grams to construct a comprehensive lexicon (26 letters + 10 numerals). The goal
of these novel constructions is to provide a lightweight alternative to text
embedding models and full TF-IDF while preserving utility for fuzzy searching.

("Vicinus" means "neighboring/nearby" in Latin.)

## Status

- [x] Hamming Distance/Similarity
- [x] Levenshtein Distance/Similarity
- [x] N-Gram Extraction
- [x] Jaccard Index
- [x] Sparse Vector Cosine Similarity
- [x] NGF (N-Gram Frequency)
- [x] NGF-IDF (N-Gram Frequency * Inverse Document Frequecy)
- [x] Optimizations: Levenshtein, NGF-IDF (reuse intermediate states)
- [x] In-Memory VectorDB: Jaccard, NGF, NGF-IDF
- [ ] Persistent VectorDB: Jaccard, NGF, NGF-IDF (sqloquent)

Open issues can be tracked [here](https://github.com/k98kurz/vicinus/issues).
Historical changes can be found in the
[changelog](https://github.com/k98kurz/vicinus/blob/master/changelog.md).

## Usage

### Installation

```bash
pip install vicinus
```

### CLI

This package includes a CLI tool for exporting an agent skill to help clankers
use this package correctly. (It also serves as decent documentation in general,
so it might also be worth scanning if you aren't coding with a clanker.)

```bash
vicinus skill               # prints to stdout
vicinus skill -o path       # exports to path/vicinus/
vicinus skill --agent       # exports to .agent/skills/vicinus/
vicinus skill --claude      # exports to .claude/skills/vicinus/
vicinus skill --codex       # exports to .agent/skills/vicinus/
vicinus skill --cursor      # exports to .cursor/skills/vicinus/
vicinus skill --opencode    # exports to .opencode/skills/vicinus/
```

The CLI can also be used to export the bundled text samples for experimentation.

```bash
vicinus samples             # prints all to stdout with filenames
vicinus samples -l          # prints list of all filenames
vicinus samples -n name     # prints a specific sample to stdout
vicinus samples -o path     # exports all files to path/
```

There is also a `vicinus version` subcommand that prints the package version.

### VectorDB

This package includes an in-memory vector database that can use Jaccard Index,
NGF, or NGF-IDF. NGF-IDF is the most accurate, but it takes significantly longer
to recalculate vectors due to the IDF component requiring recounting N-Grams
across the whole corpus and recalculating vectors for each text upon every add/
update/remove. Jaccard Index is the fastest and least accurate, but it works. NGF
seems to be a reasonable middle-of-the-road choice with better accuracy without
sacrificing too much performance, so it is the default.

The `VectorDB` class can be initialized with several options:
- `N: int`: the size of the N-Gram (default 5)
- `mode: VDBMode`: one of `VDBMode.JACCARD`, `NGF`, or `NGF_IDF` (default `NGF`)
- `idf_mode: IDFMode`: one of `IDFMode.SAVE_SPACE` or `IDFMode.SAVE_COUNTS`; valid
  only for `VDBMode.NGF_IDF` (default `SAVE_SPACE`)

```python
from vicinus import VectorDB, VDBMode

# setup
vdb = VectorDB(mode=VDBMode.NGF_IDF)
vdb.set_corpus({
    d.title_or_id: d.content
    for d in documents
}, recalculate=False) # default recalculate=True
vdb.add(title_or_id, content, recalculate=False)
vdb.remove(title_or_id) # recalculate vectors when all are added

# search
res = vdb.search(query_str, limit=2) # default limit=4; set to -1 to return all
print(res) # [(float score, title_or_id),...]
doc = vdb.get(res[0][1]) # (str content, [SparseVector|set[str]])
```

The second element returned by `VectorDB.get` will be a list with a `SparseVector`
or `set[str]` as the first element (for NGF/NGF-IDF or Jaccard, respectively) and
optionally a second `SparseVector` including the N-Gram counts for the record for
`mode=VDBMode.NGF_IDF` and `idf_mode=IDFMode.SAVE_COUNTS`.

Note that `IDFMode.SAVE_COUNTS` has thus far not shown much performance benefit in
testing, though the underlying setup mechanism was proven to be faster.

### Distance/Similarity

There are two distance metrics with corresponding similarity scores: Hamming and
Levenshtein. The computational and memory costs of a single comparison are lower
with these than going through the setup for the three vectorized mechanisms, but
repeated searches continually iterate over the same texts.

There are vectorized similarity mechanisms:
- N-Grams w/ Jaccard Index (technically a set, not a vector)
- NGF w/ Cosine Similarity
- NGF-IDF w/ Cosine Similarity

There are three other functions useful for building fuzzy search tools:
- `rank(score_fn: Callable, candidates: dict[K, V]) -> list[tuple[float, K]]`
- `select(rankings: list[tuple[float, K]], k: int = 4) -> list[tuple[float, K]]`
- `tokenize(text: str) -> list[str]`

#### Hamming

Relatively simple and cheap, but does not handle deletions/insertions. Not
suitable for fuzzy searching through large texts or many texts.

```python
from vicinus import hamming_distance, hamming_similarity, rank, select

# distance between two texts
distance = hamming_distance(text1, text2)

# rank by similarity
corpus = {1: text1, 2: text2, ...}
query = "something"
# rank all by query similarity
rankings = rank(lambda t: hamming_similarity(query, t), corpus)
# get top k=2 rankings
candidates = select(rankings, 2)
```

#### Levenshtein

More complex and robust than Hamming, but entails nested loops (polynomial time)
and some memory overhead (2x longest string). Not suitable for fuzzy searching
through large texts or many texts.

```python
from vicinus import levenshtein_distance, levenshtein_similarity, rank, select

# distance between two texts
distance = levenshtein_distance(text1, text2)

# rank by similarity
corpus = {1: text1, 2: text2, ...}
query = "something"
# rank all by query similarity
rankings = rank(lambda t: levenshtein_similarity(query, t), corpus)
# get top k=2 rankings
candidates = select(rankings, 2)
```

#### N-Grams w/ Jaccard Index

Normalize and break text into N-Grams, then use Jaccard Index on the N-Gram sets.
Scales well but loses cardinality of N-Gram frequencies -- a "bag of words"
technique. N-Grams should be at least 3 long (trigrams) but can be made longer for
larger texts. Jaccard Index is a measure of similarity based upon set inclusion of
N-Grams: shared/total (|intersection| / |union|). Only aphanumeric chars (standard
Latin lowercase + Arabic numerals) are supported.

```python
from vicinus import n_grams, jaccard_index

# calculate similarity
ng1 = n_grams(text1, N=3) # default N
ng2 = n_grams(text2, N=3)
similarity = jaccard_index(ng1, ng2) # between 0 and 1.0

# rank by similarity
corpus = {1: text1, 2: text2, ...}
query = "something"
corpus_ngrams = {k: n_grams(t) for k, t in corpus.items()}
query_ngrams = n_grams(query)
# rank all by query similarity
rankings = rank(lambda t: jaccard_index(query_ngrams, t), corpus_ngrams)
# get top k=2 rankings
candidates = select(rankings, k=2)
```

#### NGF w/ Cosine Similarity

Like TF but with N-Grams instead of terms. Includes cardinality of N-Gram
frequencies within texts but not between texts. Does not require recalculating
all vectors when a text in the corpus changes. Uses `SparseVector`s for efficiency.

```python
from vicinus import ngf, ngf_rank, ngf_select

corpus = {1: "Text 1 is some text...", 2: "Text 2 is another thing...", ...}
vecs = {k: ngf(t) for k, t in corpus.items()}
query = "text about something"
# to sort all texts
rankings = ngf_rank(query, vecs)
# or select top k=2
candidates = ngf_select(query, vecs, k=2) # or candidates = select(rankings, 2)
```

#### NGF-IDF w/ Cosine Similarity

Like TF-IDF but with N-Grams instead of terms. Considers N-Gram frequencies within
texts and between texts. Requires setup phase that recalculates the IDF for a
corpus whenever a text changes.

```python
from vicinus import ngf_idf_setup, ngf_idf_rank, ngf_idf_select

# initial setup
corpus = {1: "Text 1 is some text...", 2: "Text 2 is another thing...", ...}
corpus_idf, vecs, ngcs = ngf_idf_setup(corpus, N=3) # default N

# add another text and recalculate, reusing n-gram counts
corpus[new_id] = new_text
corpus_idf, vecs, ngcs = ngf_idf_setup(corpus, N=3, ng_counts=ngcs)

query = "text about something"
# to sort all texts by cosine similarity
all_ranked = ngf_idf_rank(query, corpus_idf, vecs, N=3) # default N
# or select the top k=2
candidates = ngf_idf_select(query, corpus_idf, vecs, 2, N=3)
```

Note: N=5 resulted in far more accurate rankings for the test vectors than the
default N=3 for Jaccard Index, NGF, and NGF-IDF. This may require some tuning.
Higher `N` values lead to sparser vectors and lower cosine similarity/Jaccard
Index scores, but this appears to be primarily a culling of noise.

#### SparseVector

The `SparseVector` class stores non-0 values and their indices. It implements several
noteworth methods:
- `get(self, index: int, default = 0) -> int|float`
- `keys(self) -> set[int]`
- `values(self) -> ValuesView[int|float]`
- `items(self) -> ItemsView[tuple[int, int|float]]`
- `norm(self) -> float`
- `dot_product(self, other: SparseVector) -> float`
- `cosine_similarity(self, other: SparseVector) -> float`
- `copy(self) -> SparseVector`

### Docs

Documentation generated by [autodox](https://pypi.org/project/autodox)
can be found [here](https://github.com/k98kurz/vicinus/blob/master/docs.md).

## Note on Generative AI Use

All code and documentation (except docs.md) was written by hand, including the
test suite. I used mistral and gemma4:e4b (running on my own machine via ollama
with some fun bashfu) to write the text samples for testing (they are slop). I
also used gemma4:31b via Ollama cloud in OpenCode for code review (it caught a
few loose ends and documentation issues).

The inspiration for this library came during development of a yet unreleased new
agentic harness system that I intend to optimize for small, local models: as I
drafted a spec for a `tool_search` tool as an option to avoid context bloat, I
initially thought to copy a Hamming distance implementation I wrote a few months
ago in a different project for fuzzy search. I reviewed the draft spec with
gemma4:e4b, and it suggested I look at Levenshtein distance and Jaccard Index.
I then read some formulas on Wikipedia, implemented Levenshtein distance, N-grams,
and Jaccard index, and then derived the concept of NGF/NGF-IDF independently while
reflecting upon the code I had just written and the inherent difficulties with
TF/TF-IDF (i.e. the need to construct an indexed lexicon or use a text embedding
model).

Code projects like this are too fun to give to clankers. Gotta maintain some joy
in life.

## Testing

First, clone the repo, set up a virtual env, and install requirements from
requirements.txt. Then run the following command to run the test suite.

```bash
pip install -e . # or use uv
python -m unittest discover tests
```

There are currently 46 tests covering all functionality except the CLI, which is
tested manually. Note that one of the tests occasionally fails due to floating
point arithmetic behaviors, and one asserting better performance for
`IDFMode.SAVE_COUNTS` occasionally fails, perhaps due to unpredictable memory
allocation patterns; opportunities for further optimizations will be explored in
the future.

## Contributing

Check out the [Pycelium discord server](https://discord.gg/b2QFEJDX69). If you
experience a problem, please discuss it on the Discord server. All suggestions
for improvement are also welcome, and the best place for that is also Discord.
If you experience a bug and do not use Discord, open an issue or discussion on
Github.

## ISC License

Copyright (c) 2026 Jonathan Voss (k98kurz)

Permission to use, copy, modify, and/or distribute this software
for any purpose with or without fee is hereby granted, provided
that the above copyright notice and this permission notice appear in
all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
