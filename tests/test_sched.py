import pytest

import collections

from kerosene import optimizer, sched, shape


class MockOptimizer(object):
    def __init__(self, n_layers):
        self.param_groups = [collections.Counter() for _ in range(n_layers)]


def test_clr_schedule_no_momentum():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    optim.set_lrs([1, 2])
    optim.set_momentums(0.95)
    schedule = sched.clr(optim, 4, lr_factor=2)

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
    schedule = sched.clr(optim, 4, lr_factor=2, momentums=(3/4, 1/4))

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


def test_one_cycle_schedule():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    optim.set_lrs([1, 2])
    optim.set_momentums(0.95)
    schedule = sched.one_cycle(
        optim,
        5,
        lr_factor=2,
        momentums=(3/4, 1/4),
        anneal_share=1/5,
        anneal_factor=16,
    )

    schedule.init_training()
    assert list(optim.get_lrs()) == pytest.approx([1/2, 1])
    assert list(optim.get_momentums()) == pytest.approx([3/4, 3/4])

    schedule.step()
    assert list(optim.get_lrs()) == pytest.approx([3/4, 3/2])
    assert list(optim.get_momentums()) == pytest.approx([1/2, 1/2])

    schedule.step()
    assert list(optim.get_lrs()) == pytest.approx([1, 2])
    assert list(optim.get_momentums()) == pytest.approx([1/4, 1/4])

    schedule.step()
    assert list(optim.get_lrs()) == pytest.approx([3/4, 3/2])
    assert list(optim.get_momentums()) == pytest.approx([1/2, 1/2])

    schedule.step()
    assert list(optim.get_lrs()) == pytest.approx([1/2, 1])
    assert list(optim.get_momentums()) == pytest.approx([3/4, 3/4])

    schedule.step()
    assert list(optim.get_lrs()) == pytest.approx([1/32, 1/16])
    assert list(optim.get_momentums()) == pytest.approx([3/4, 3/4])


def test_nop_is_fine():
    schedule = sched.nop()
    schedule.init_training()
    for _ in range(4):
        schedule.step()


def test_not_yet_programmable_parameter():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    optim.set_lrs([1, 2])
    optim.set_momentums(0.95)
    with pytest.raises(NotImplementedError):
        schedule = sched.Schedule(optim, 4, {'rando': shape.line(1/4, 3/4)})
        schedule.init_training()


def test_step_past_finish_fails():
    optim = optimizer.ProgrammableOptimizer(MockOptimizer(2))
    schedule = sched.Schedule(optim, 4, {})
    schedule.init_training()
    for _ in range(4):
        schedule.step()
    with pytest.raises(ValueError):
        schedule.step()
