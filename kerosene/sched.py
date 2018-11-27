from . import shape, util


class Schedule(object):
    """Steppable schedule for programmable hyperparameters.

    Use the given shapes for each parameter to drive the updates for a
    programmable optimizer.

    This is still using a custom counter to track steps within a cycle,
    it would be better to inject that value as a method parameter.
    """

    def __init__(self, optim, nb, param_shapes):
        self.nb = nb
        self.optim = optim
        self.init_lrs = None if optim is None else optim.get_lrs()
        self.param_shapes = param_shapes

    def init_training(self):
        self.iter = 0
        self._set_params()

    def step(self):
        if self.nb > 0 and self.iter >= self.nb:
            raise ValueError(f'already iterated past {self.nb}')
        self.iter += 1
        self._set_params()

    def _set_params(self):
        for param, shape_fn in self.param_shapes.items():
            interp = shape_fn(self.iter / self.nb)
            if param == 'lr':
                self.optim.set_lrs(self.init_lrs * interp)
            elif param == 'momentum':
                self.optim.set_momentums(interp)
            elif param == 'wd':
                self.optim.set_wds(interp)
            else:
                raise NotImplementedError(f'unsupported param: {param}')


def nop():
    return Schedule(None, 0, {})


def clr(optim, nb, lr_factor=10, momentums=None, wds=None):
    return Schedule(optim, nb, _filter_nones({
        'lr': shape.clr(lr_factor),
        'momentum': _tuple_shape(momentums),
        'wd': _tuple_shape(wds),
    }))


def stlr(optim, nb, lr_factor=10, up_share=1/4, momentums=None, wds=None):
    return Schedule(optim, nb, _filter_nones({
        'lr': shape.stlr(lr_factor, up_share),
        'momentum': _tuple_shape(momentums),
        'wd': _tuple_shape(wds),
    }))


def burn_in(optim, nb, lr_factor=10, up_share=1/10, momentum=None, wd=None):
    return Schedule(optim, nb, _filter_nones({
        'lr': shape.burn_in(lr_factor, up_share),
        'momentum': None if momentum is None else shape.const(momentum),
        'wd': None if wd is None else shape.const(wd),
    }))


def one_cycle(
    optim,
    nb,
    lr_factor=10,
    momentums=(0.95, 0.85),
    anneal_share=1/10,
    anneal_factor=100,
    wds=None,
):
    return Schedule(optim, nb, _filter_nones({
        'lr': shape.one_cycle(lr_factor, anneal_share, anneal_factor),
        'momentum': shape.one_cycle_momentum(momentums[0], momentums[1], anneal_share),
        'wd': _tuple_shape(wds),
    }))


def _filter_nones(m):
    return {k: v for k, v in m.items() if v is not None}


def _tuple_shape(vals):
    if vals is None:
        return None
    if util.is_listy(vals):
        return shape.triangle(*vals)
    return shape.const(vals)
