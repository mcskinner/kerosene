import contextlib
import numpy as np
import torch

from distutils.version import LooseVersion
from torch.autograd import Variable

from . import util


IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()


def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if USE_GPU else x


def variable(x, requires_grad=False):
    if type(x) == Variable:
        return x
    if util.is_listy(x):
        return [variable(o, requires_grad) for o in x]
    return Variable(tensor(x), requires_grad=requires_grad)


def tensor(x, cuda=True):
    x = create_tensor(x)
    return to_gpu(x, async=True) if cuda else x


def create_tensor(x):
    if torch.is_tensor(x):
        return x

    x = np.ascontiguousarray(x)
    if x.dtype in (np.int8, np.int16, np.int32, np.int64):
        return torch.LongTensor(x.astype(np.int64))
    if x.dtype in (np.float32, np.float64):
        return torch.FloatTensor(x)
    raise NotImplementedError(x.dtype)


def no_grad():
    return torch.no_grad() if IS_TORCH_04 else contextlib.suppress()
