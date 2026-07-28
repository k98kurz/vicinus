def type_assert(condition: bool, message: str = 'invalid type') -> None:
    if not condition:
        raise TypeError(message)


def value_assert(condition: bool, message: str = 'invalid value') -> None:
    if not condition:
        raise ValueError(message)
