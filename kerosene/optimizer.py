import numpy as np

from . import util


def make(opt_fn, layer_groups, lrs, wds=0):
    """Create a programmable optimizer based on the input config.

    An optimizer produced by opt_fn will be used to update model params,
    which are split up into different layer_groups. That can also just
    be the entire model. The optimizer is then initialized with lr and
    wds set on a per-group basis, with broadcasting for atomic values.
    """

    layer_groups = util.listify(layer_groups)
    return ProgrammableOptimizer(opt_fn([
        {
            'params': _trainable_params(layer_group),
            'lr': lr,
            'wd': wd,
        }
        for layer_group, lr, wd in zip(
            layer_groups,
            util.list_along(lrs, layer_groups),
            util.list_along(wds, layer_groups),
        )
    ]))


def _trainable_params(group):
    return [p for layer in util.listify(group) for p in layer.parameters() if p.requires_grad]


class ProgrammableOptimizer(object):
    """Make it easy to program the parameters of a PyTorch optimizer.

    The parameters here are well suited for automation via the shape package.
    See sched.CLR for an example usage.
    """

    def __init__(self, optim):
        self.optim = optim
        self.params = optim.param_groups[0].keys()

    def zero_grad(self):
        """Convenient pass-through to the inner optimizer."""
        self.optim.zero_grad()

    def step(self):
        """Apply weight decay, then pass- hrough to the inner optimizer.

        This is the idea proposed in AdamW. This has a similar effect on
        weights as if they were just added to the loss, but without
        messing with adaptive gradient methods like RMSProp or Adam.
        """

        if 'wd' in self.params:
            self._apply_weight_decay()
        self.optim.step()

    def _apply_weight_decay(self):
        for group in self.optim.param_groups:
            decay = group['lr'] * group['wd']
            if decay == 0:
                continue
            for param in group['params']:
                if param.grad is not None:
                    param.data.sub_(decay)

    def get_param(self, key):
        return np.array([pg[key] for pg in self.optim.param_groups])

    def set_param(self, key, vals, index=None):
        vals = util.list_along(vals, self.optim.param_groups)
        for params, val in zip(self.optim.param_groups, vals):
            if index is None:
                params[key] = val
            else:
                old_val = list(params[key])
                old_val[index] = val
                params[key] = tuple(old_val)

    def get_lrs(self):
        return self.get_param('lr')

    def get_momentums(self):
        return self.get_param('momentum')

    def set_lrs(self, lrs):
        self.set_param('lr', lrs)

    def set_wds(self, wds):
        """New-style weight decay which acts directly on the weights."""
        self.set_param('weight_decay', 0)
        self.set_param('wd', wds)

    def set_wds_broken(self, wds):
        """Old-style weight decay in the loss, which breaks e.g. Adam."""
        self.set_param('wd', 0)
        self.set_param('weight_decay', wds)

    def set_momentums(self, momentums):
        if 'betas' in self.params:
            self.set_param('betas', momentums, index=0)
        else:
            self.set_param('momentum', momentums)

    def set_betas(self, betas):
        if 'betas' in self.params:
            self.set_param('betas', betas, index=1)
        elif 'alpha' in self.params:
            self.set_param('alpha', betas)
        else:
            raise ValueError(f'{self.optim.__name__} has no betas to set')
