import pytest

import collections

from kerosene import optimizer


class MockOptimizer(object):
    def __init__(self, n_layers):
        self.param_groups = [collections.Counter() for _ in range(n_layers)]


def test_get_and_set_learning_rates():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(3))
    assert list(optim.get_lrs()) == [0, 0, 0]

    optim.set_lrs(1)
    assert list(optim.get_lrs()) == [1, 1, 1]

    optim.set_lrs([1e-2, 1e-1, 1])
    assert list(optim.get_lrs()) == [1e-2, 1e-1, 1]


def test_get_and_set_momentums():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    assert list(optim.get_momentums()) == [0, 0]

    optim.set_momentums(1)
    assert list(optim.get_momentums()) == [1, 1]

    optim.set_momentums([1/3, 1])
    assert list(optim.get_momentums()) == [1/3, 1]


def test_failure_on_bad_parameter_lengths():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(3))

    with pytest.raises(ValueError):
        optim.set_lrs([1, 2])

    with pytest.raises(ValueError):
        optim.set_momentums([3, 4])
