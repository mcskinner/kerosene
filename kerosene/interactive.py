import sys
import tqdm as tq

from ipykernel.kernelapp import IPKernelApp


def in_notebook():
    return IPKernelApp.initialized()


def in_ipynb():
    try:
        cls = get_ipython().__class__.__name__
        return cls == 'ZMQInteractiveShell'
    except NameError:
        return False


def clear_tqdm():
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst:
        return
    try:
        for i in range(len(inst)):
            inst.pop().close()
    except Exception:
        pass


def tqdm(*args, **kwargs):
    if in_notebook():
        clear_tqdm()
        return tq.tqdm(*args, file=sys.stdout, **kwargs)
    return tq.tqdm(*args, **kwargs)


def trange(*args, **kwargs):
    if in_notebook():
        clear_tqdm()
        return tq.trange(*args, file=sys.stdout, **kwargs)
    return tq.trange(*args, **kwargs)


def tnrange(*args, **kwargs):
    if in_notebook():
        return tq.tnrange(*args, **kwargs)
    return tq.trange(*args, **kwargs)
