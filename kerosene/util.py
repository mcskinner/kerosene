import collections


def is_listy(x):
    return isinstance(x, (list, tuple))


def is_iter(x):
    return isinstance(x, collections.Iterable)


def listify(x):
    return x if is_listy(x) else [x]


def list_along(x, to_match):
    """Make x into a list that is as long as to_match."""

    x = x if is_iter(x) else [x]
    n = to_match if isinstance(to_match, int) else len(to_match)
    if len(x) == 1:
        return x * n
    if len(x) != n:
        raise ValueError(f'x and to_match should be the same size, was {len(x)} and {n}')
    return x
