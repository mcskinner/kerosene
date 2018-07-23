import torch

from . import sched, util


class Manager():
    """Manage the batch execution lifecycle for training and validation.

    After inputs are run through the model, loss is computed.
    If requested, we run backpropagation and take an optimization step,
    as well as a scheduler step which may update learning rate or other
    optimization parameters.

    If requested, typically during evaluation, we also compute some
    additional metrics over the batch.

    See loop.fit for usage.
    """

    def __init__(self, model, optim, loss, schedule=None, metrics=None, seq_first=False):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.schedule = schedule or sched.nop()
        self.metrics = metrics or []
        self.seq_first = seq_first

    def init_training(self):
        self.schedule.init_training()

    def step(self, xs, y, with_step=False):
        preds = self.model(*xs)
        loss = self.loss(preds, y)
        if with_step:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.schedule.step()
        return _item(loss.data), preds

    def train_runner(self):
        _set_train(self.model)
        return self._make_runner(True, False)

    def eval_runner(self):
        self.model.eval()
        return self._make_runner(False, True)

    def _make_runner(self, with_step=False, with_metrics=False, ema_window=50):
        if hasattr(self.model, 'reset'):
            self.model.reset()
        metric_fns = self.metrics if with_metrics else []
        return TrackedRunner(self, with_step, metric_fns, ema_window)


def _item(x):
    return x.item() if hasattr(x, 'item') else x[0]


class TrackedRunner(object):
    """Track average loss and metrics across the lifetime of a runner."""

    def __init__(self, mgr, with_step, metric_fns, ema_window=50):
        self.mgr = mgr
        self.with_step = with_step
        self.metric_fns = metric_fns
        self.avg_loss = WeightedAverage()
        self.avg_metrics = [WeightedAverage() for _ in metric_fns]
        self.running_loss = ExponentialMovingAverage(ema_window)

    def run(self, xs, y):
        loss, preds = self.mgr.step(xs, y, self.with_step)
        metrics = [fn(preds.data, y.data) for fn in self.metric_fns]
        x0 = xs[0] if util.is_listy(xs) else xs
        batch_sz = x0.shape[1 if self.mgr.seq_first else 0]
        self.avg_loss.update(loss, wt=batch_sz)
        for average, metric in zip(self.avg_metrics, metrics):
            average.update(metric, wt=batch_sz)
        return self.running_loss.update(loss)

    def report(self):
        return self.avg_loss.value(), [metric.value() for metric in self.avg_metrics]


class ExponentialMovingAverage(object):
    """Stateful encapsulation of a de-biased exponential moving average."""

    def __init__(self, window):
        if window < 1:
            raise ValueError(f'window must be at least 1, was {window}')
        self.momentum = 1 - 1/window
        self.average = 0
        self.debias = 0

    def update(self, val):
        x = self.momentum
        self.average = x*self.average + (1-x)*val
        self.debias = x*self.debias + (1-x)
        return self.value()

    def value(self):
        return self.average / self.debias


class WeightedAverage(object):
    """Stateful encapsulation of a weighted average."""

    def __init__(self):
        self.numer = 0
        self.denom = 0

    def update(self, val, wt=1):
        if wt < 0:
            raise ValueError(f'weight must not be negative, was {wt}')
        self.numer += wt*val
        self.denom += wt
        return self.value()

    def value(self):
        return self.numer / self.denom


def _set_train(model):
    children = model if util.is_listy(model) else list(model.children())
    for child in children:
        _set_train(child)
    if isinstance(model, torch.nn.Module):
        if _is_frozen(model):
            model.eval()
        else:
            model.train()


def _is_frozen(model):
    if hasattr(model, 'p') and 'drop' in type(model).__name__.lower():
        return getattr(model, 'drop_freeze', False)
    if hasattr(model, 'running_mean'):
        return getattr(model, 'bn_freeze', False) or not getattr(model, 'trainable', False)
    return False
