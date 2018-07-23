import numpy as np

from . import batches, optimizer, sched, torch_util
from .interactive import tnrange, tqdm


def fit_and_finish(
    model, opt_fn, loss_fn,
    trn_dl, val_dl, n_epochs, lr,
    metrics=None, seq_first=False
):
    n_batches = n_epochs * len(trn_dl)
    optim = optimizer.make(opt_fn, model, lr)
    schedule = sched.CLR(optim, n_batches, momentums=(0.95, 0.85))
    mgr = batches.Manager(model, optim, loss_fn, schedule, metrics, seq_first)
    return fit(mgr, trn_dl, val_dl, n_epochs)


def fit(mgr, trn_dl, val_dl, n_epochs):
    mgr.init_training()
    for epoch in tnrange(n_epochs, desc='Epoch'):
        train_loss, _ = run_epoch(mgr.train_runner(), trn_dl)

        with torch_util.no_grad():
            val_loss, val_metrics = run_epoch(mgr.eval_runner(), val_dl, track_progress=False)

        if epoch == 0:
            _print_names(mgr.metrics)
        _print_stats(epoch, [train_loss] + [val_loss] + val_metrics)

    return val_loss, val_metrics


def run_epoch(runner, dl, track_progress=True):
    batches = tqdm(dl, leave=False, total=len(dl), miniters=0) if track_progress else dl
    for *x, y in batches:
        loss = runner.run(torch_util.variable(x), torch_util.variable(y))
        if track_progress:
            batches.set_postfix(loss=loss, refresh=False)
    return runner.report()


def _print_names(metrics):
    _print_wide(['epoch', 'trn_loss', 'val_loss'] + [fn.__name__ for fn in metrics])


def _print_stats(epoch, values, decimals=6):
    _print_wide([epoch] + list(np.round(values, decimals)), ralign_first=True)


def _print_wide(vals, ralign_first=False, decimals=6):
    first = '{!s:^10}' if ralign_first else '{!s:10}'
    layout = first + ' {!s:10}' * (len(vals)-1)
    print(layout.format(*vals))
