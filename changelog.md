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
