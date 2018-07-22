import collections

from kerosene import optimizer, sched


class MockOptimizer(object):
    def __init__(self, n_layers):
        self.param_groups = [collections.Counter() for _ in range(n_layers)]


def test_clr_schedule_no_momentum():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    optim.set_lrs([1, 2])
    optim.set_momentums(0.95)
    schedule = sched.CLR(optim, 4, lr_factor=2)

    schedule.init_training()
    assert list(optim.get_lrs()) == [1/2, 1]
    assert list(optim.get_momentums()) == [0.95, 0.95]

    schedule.step()
    assert list(optim.get_lrs()) == [3/4, 3/2]
    assert list(optim.get_momentums()) == [0.95, 0.95]

    schedule.step()
    assert list(optim.get_lrs()) == [1, 2]
    assert list(optim.get_momentums()) == [0.95, 0.95]

    schedule.step()
    assert list(optim.get_lrs()) == [3/4, 3/2]
    assert list(optim.get_momentums()) == [0.95, 0.95]

    schedule.step()
    assert list(optim.get_lrs()) == [1/2, 1]
    assert list(optim.get_momentums()) == [0.95, 0.95]


def test_clr_schedule_with_momentum():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    optim.set_lrs([1, 2])
    optim.set_momentums(0.95)
    schedule = sched.CLR(optim, 4, lr_factor=2, momentums=(3/4, 1/4))

    schedule.init_training()
    assert list(optim.get_lrs()) == [1/2, 1]
    assert list(optim.get_momentums()) == [3/4, 3/4]

    schedule.step()
    assert list(optim.get_lrs()) == [3/4, 3/2]
    assert list(optim.get_momentums()) == [1/2, 1/2]

    schedule.step()
    assert list(optim.get_lrs()) == [1, 2]
    assert list(optim.get_momentums()) == [1/4, 1/4]

    schedule.step()
    assert list(optim.get_lrs()) == [3/4, 3/2]
    assert list(optim.get_momentums()) == [1/2, 1/2]

    schedule.step()
    assert list(optim.get_lrs()) == [1/2, 1]
    assert list(optim.get_momentums()) == [3/4, 3/4]
