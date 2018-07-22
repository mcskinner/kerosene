import pytest

import itertools

from kerosene import util


def test_is_listy():
    assert util.is_listy([1])
    assert util.is_listy([1, 2])
    assert util.is_listy((3, 4))
    assert not util.is_listy(1)
    assert not util.is_listy('abc')
    assert not util.is_listy(itertools.count())


def test_is_iter():
    assert util.is_iter([1])
    assert util.is_iter([1, 2])
    assert util.is_iter((3, 4))
    assert util.is_iter('abc')
    assert util.is_iter(itertools.count())
    assert not util.is_iter(1)


def test_listify():
    assert util.listify(1) == [1]
    assert util.listify([1, 2]) == [1, 2]
    assert util.listify((3, 4)) == (3, 4)
    assert util.listify('abc') == ['abc']


def test_list_along():
    assert util.list_along(1, [1]) == [1]
    assert util.list_along(1, [1, 2]) == [1, 1]
    assert util.list_along(1, 3) == [1, 1, 1]
    assert util.list_along((3, 4), [1, 2]) == (3, 4)
    assert util.list_along((3, 4, 5), 3) == (3, 4, 5)


def test_list_along_failures():
    with pytest.raises(ValueError):
        util.list_along((3, 4, 5), [1, 2])

    with pytest.raises(ValueError):
        util.list_along((3, 4, 5), 5)
