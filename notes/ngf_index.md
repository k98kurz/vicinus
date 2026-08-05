# NGF Index

For indexing into the `SparseVector`, each N-Gram is parsed by assigning numeric
values to alphanumeric chars: 0-9 are assigned 0-9; a-z are assigned 10-35. Each
char in the N-Gram is so converted then multiplied by a base B raised to `(N-i-1)`,
where `i` is the index of the char in the N-Gram, i.e. `val(ng[i]) * B ** (N-i-1)`.
(Double asterisk denotes exponentiation.) The index of the N-Gram is then the sum
of these products.

There was an error in the selection of the base in v0.0.1 and v0.0.2: instead of
36, the base was set to 10. Below is the offending code:

```python
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
```

For v0.0.3, I corrected it to have the proper base:

```python
def ngf_index(ng: str) -> int:
    """Calculate the NGF vector index for a given alphanumeric N-Gram."""
    N = len(ng)
    ind = 0
    for i in range(N):
        c = ng[i]
        if '0' <= c <= '9':
            ind += (ord(c) - ord('0')) * (36 ** (N-i-1))
        elif 'a' <= c <= 'z':
            ind += (10 + ord(c) - ord('a')) * (36 ** (N-i-1))
    return ind
```

This small error severely impacted the signal-to-noise ratio of cosine similarity
in both NGF and NGF-IDF. When I discovered this, I ran the test suite to get some
performance metrics, then I fixed it and ran them again. Both NGF and NGF-IDF
improved substantially. (Jaccard was unaffected since it does not use vectors.)
The results below show the top cosine similarity score from a search against a
VectorDB as well as the ratio between the top score and the 2nd best score:

| Algo    | Top Score (B=10) | Top Score (B=36) | Ratio (B=10) | Ratio (B=36) |
|---------|------------------|------------------|--------------|--------------|
| NGF     | 0.31373          | 0.30864          | 15.93        | 18.36        |
| NGF-IDF | 0.32093          | 0.37995          | 29.43        | 47.87        |
