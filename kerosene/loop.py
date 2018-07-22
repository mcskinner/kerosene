import numpy as np
import torch


from . import batches, optimizer, sched, torch_util


def fit_and_finish(model, opt_fn, loss_fn, trn_dl, val_dl, n_epochs, lr):
    n_batches = n_epochs * len(trn_dl)
    optim = optimizer.make(opt_fn, model, lr)
    schedule = sched.CLR(optim, n_batches, momentums=(0.95, 0.85))
    mgr = batches.Manager(model, optim, loss_fn, schedule)
    return fit(mgr, trn_dl, val_dl, n_epochs)


def fit(mgr, trn_dl, val_dl, n_epochs):
    mgr.init_training()
    for epoch in range(n_epochs):
        mgr.set_train()
        train_loss, _ = run_epoch(mgr.train_runner(), trn_dl)

        mgr.set_eval()
        with torch.no_grad():  # Assumes PyTorch 0.4
            val_loss, val_metrics = run_epoch(mgr.eval_runner(), val_dl)

        if epoch == 0:
            _print_names(mgr.metrics)
        _print_stats(epoch, [train_loss] + [val_loss] + val_metrics)

    return val_loss, val_metrics


def run_epoch(runner, dl):
    tracked = runner.tracked()
    for *x, y in batches:
        tracked.run(torch_util.variable(x), torch_util.variable(y))
    return tracked.report()


def _print_names(metrics):
    _print_wide(['epoch', 'trn_loss', 'val_loss'] + [fn.__name__ for fn in metrics])


def _print_stats(epoch, values, decimals=6):
    _print_wide([epoch] + list(np.round(values, decimals)), ralign_first=True)


def _print_wide(vals, ralign_first=False, decimals=6):
    first = '{!s:^10}' if ralign_first else '{!s:10}'
    layout = first + ' {!s:10}' * (len(vals)-1)
    print(layout.format(*vals))
