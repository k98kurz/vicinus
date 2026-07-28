def tokenize(text: str) -> list[str]:
    """Preprocess some text by casting to lower, removing punctuation,
        and splitting on word boundaries. Returns only alphanumeric
        tokens.
    """
    return [
        ''.join(c for c in word if c.isalnum())
        for word in text.lower().split()
    ]
