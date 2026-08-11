## 0.0.3

- Added new CTF and CTF-IDF mechanisms: CT = CRC32(Token)
  - Essentially TF/TF-IDF with deterministic indices and tokenizer preprocessing
  - Can use full `2**32` dimensions of crc32 or smaller; default is 10,000
  - Integrated into `VectorDB` via `VDBMode.CTF` and `VDBMode.CTF_IDF`
- Added `SparseVector.__add__` and `SparseVector.__eq__` methods for convenience
- Improved performance of IDF vector computation from `O(D*|V|^2)` to `O(D*|V|)`
  - Speed-up from reusing intermediate counts vectors decreased to ~0-9%
- Improved `VectorDB.__init__` input validation to fail-fast on bad `N` values
- Bug fixes:
  - `levenshtein_distance` had an off-by-one error in state space dimensionality
  - `ngf_index` had the wrong base, causing index collisions; improved similarity
    score SNRs in test vectors by 15% for NGF and 62% for NGF-IDF
  - `tokenize` was returning empty tokens when a word contained only non-alnum chars
- Misc: slightly improved test coverage; improved docstrings; fixed some annotations

## 0.0.2

- Optimized Levenshtein Distance: converted from recursive function w/ memoization to
  a nested loop
- Refactored `ngf_idf_setup` to accept intermediate states for faster recomputation
  - Can now pass `ng_counts: dict[Any, SparseVector]` to avoid recounting for all
    texts; will still perform counting for new texts; entails 10-17%
    speed improvement (highly variable in testing); also tested reusing ngf vectors,
    but it had no benefit
  - Now also returns `ng_counts` for use in future calls
- Updated `rank` to accept `candidates: dict` instead of `candidates: list`
- New `SparseVector.copy()` method that returns a copy of the vector
- New `VectorDB` in-memory vector database class
  - `VDBMode` enum with members `JACCARD`, `NGF`, and `NGF_IDF`
    - `VDBMode.JACCARD` is the cheapest/fastest to set up (20ms) and query (2ms)
    - `VDBMode.NGF` is more costly (200ms, 20ms) but substantially better
    - `VDBMode.NGF_IDF` setup is extremely costly/slow (10s) but the most accurate;
      same query times as `VDBMode.NGF`
  - `IDFMode` enum with members `SAVE_SPACE` and `SAVE_COUNTS`
  - In practice, `IDFMode.SAVE_COUNTS` does not consistently save time w/
    `VectorDB.recalculate()`; savings seem to be within the margin of error; test
    asserting better performance occasionally fails


## 0.0.1

- Initial version:
  - Hamming & Levenshtein distance and similarity
  - N-Grams w/ Jaccard Index
  - NGF w/ Cosine Similarity (and `SparseVector`s)
  - NGF-IDF w/ Cosine Similarity
  - CLI: export skill and bundled samples
