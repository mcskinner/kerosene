import pytest

import numpy as np

from kerosene import batches


def test_exponential_moving_average():
    ema = batches.ExponentialMovingAverage(2)
    assert ema.update(1) == (1/2) / (1/2)
    assert ema.update(2) == (1/4 + 1) / (3/4)
    assert ema.update(3) == (1/8 + 1/2 + 3/2) / (7/8)


def test_keep_last_ema():
    ema = batches.ExponentialMovingAverage(1)
    assert ema.update(1) == 1
    assert ema.update(2) == 2
    assert ema.update(3) == 3


def test_bad_ema_windows():
    with pytest.raises(ValueError):
        batches.ExponentialMovingAverage(-1)

    with pytest.raises(ValueError):
        batches.ExponentialMovingAverage(0)

    with pytest.raises(ValueError):
        batches.ExponentialMovingAverage(1/2)


def test_weighted_average():
    avg = batches.WeightedAverage()
    assert avg.update(1) == 1 / 1
    assert avg.update(2, wt=2) == (1+4) / (1+2)
    assert avg.update(3, wt=3) == (1+4+9) / (1+2+3)


def test_bad_average_weights():
    avg = batches.WeightedAverage()
    with pytest.raises(ValueError):
        avg.update(1, -1)


def test_tracked_runner():
    runner = batches.TrackedRunner(MockManager(), False, [accuracy, mean_err], ema_window=2)

    assert runner.run(array([1]), Var(2)) == (1/2) / (1/2)
    assert runner.run(array([2]), Var(4)) == (1/4 + 1) / (3/4)
    assert runner.run(array([3]), Var(6)) == (1/8 + 1/2 + 3/2) / (7/8)

    loss, metrics = runner.report()
    assert loss == 2
    assert metrics == [0, 4]


def accuracy(lhs, rhs):
    return np.mean(lhs == rhs)


def mean_err(lhs, rhs):
    return np.sum(np.abs(lhs - rhs))


def array(x):
    return np.array([x])


class Var(object):
    def __init__(self, val):
        self.data = array(val)


class MockManager(object):
    def __init__(self):
        self.metric_fns = [accuracy]
        self.seq_first = False

    def step(self, xs, y, with_step):
        return xs, Var(0)
