def hamming_distance(str1: str, str2: str, normalize: bool = False) -> int | float:
    """Calculate Hamming distance between two strs. If `normalize` is
        `False` (default), distance is returned directly as an int. If
        `normalize` is set to `True`, the distance is normalized to a
        float proportional to the length of the longest string.
    """
    if len(str1) > len(str2):
        ss = str1
        str1 = str2
        str2 = ss
    diff = len(str2) - len(str1)
    diff += sum([0 if str1[i] == str2[i] else 1 for i in range(len(str1))])
    if normalize:
        return diff / max(len(str1), len(str2))
    return diff

def hamming_similarity(str1: str, str2: str) -> float:
    """Hamming similarity: 1 - normalized Hamming distance."""
    return 1.0 - hamming_distance(str1, str2, True)
