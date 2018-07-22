import abc

from . import shape, util


class Schedule(abc.ABC):
    @abc.abstractmethod
    def init_training(self):
        pass

    @abc.abstractmethod
    def step():
        pass


class NopSchedule(Schedule):
    """Do-nothing schedule, as an alternative to None."""

    def init_training(self):
        pass

    def step(self):
        pass


class CLR(Schedule):
    """Steppable CLR schedule for learning rate and momentum.

    We build out shapes for each parameter, and use those to drive the
    updates for a programmable optimizer.

    This is still using a custom counter to track steps within a cycle,
    of which there can be multiple. It would probably be better to
    inject that value as a method parameter.
    """

    def __init__(self, optim, nb, lr_factor=10, momentums=None):
        self.nb = nb
        self.optim = optim
        self.init_lrs = optim.get_lrs()
        self.lr_shape = shape.clr(lr_factor)

        if momentums is None:
            self.momentum_shape = None
        elif util.is_listy(momentums):
            mom0, mom1 = momentums
            self.momentum_shape = shape.triangle(mom0, mom1)
        else:
            self.momentum_shape = shape.const(momentums)

    def init_training(self):
        self.cycle_iter, self.cycle_count = 0, 0
        self._set_params()

    def step(self):
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.cycle_count += 1
        self._set_params()

    def _set_params(self):
        self.optim.set_lrs(self.init_lrs * self.lr_shape(self.cycle_iter / self.nb))
        if self.momentum_shape is not None:
            self.optim.set_momentums(self.momentum_shape(self.cycle_iter / self.nb))
