import abc


class Shape(abc.ABC):
    """A function from [0,1]->[lo,hi], e.g. linear interpolation."""

    @abc.abstractmethod
    def compute(self, x):
        pass

    @abc.abstractmethod
    def range(self):
        pass

    def __call__(self, x):
        if x < 0 or x > 1:
            raise ValueError('x must be in [0, 1], was {}'.format(x))
        return self.compute(x)


class ConstantShape(Shape):
    """A constant shape of 1, i.e. the identity transformation."""

    def compute(self, x):
        return 1

    def range(self):
        return (1, 1)


class LineShape(Shape):
    """The shape of a linear progression from 0 to 1."""

    def compute(self, x):
        return x

    def range(self):
        return (0, 1)


class Shift(Shape):
    """Some other shape, additively shifted by some amount."""

    def __init__(self, shape, shift):
        self.shape = shape
        self.shift = shift

    def compute(self, x):
        return self.shift + self.shape(x)

    def range(self):
        lo, hi = self.shape.range()
        return (self.shift+lo, self.shift+hi)


class Scale(Shape):
    """Some other shape, multiplied by some scale."""

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def compute(self, x):
        return self.scale * self.shape(x)

    def range(self):
        lo, hi = self.shape.range()
        if self.scale < 0:
            lo, hi = hi, lo
        return (self.scale*lo, self.scale*hi)


class Chain(Shape):
    """Compose two shapes with some sharing of the epochs, default 50/50."""

    def __init__(self, shape0, shape1, first_share=0.5):
        self.shape0 = shape0
        self.shape1 = shape1
        self.first_share = first_share

    def compute(self, x):
        if x < self.first_share:
            return self.shape0(x / self.first_share)
        return self.shape1((x-self.first_share) / (1-self.first_share))

    def range(self):
        lo0, hi0 = self.shape0.range()
        lo1, hi1 = self.shape1.range()
        return (min(lo0, lo1), max(hi0, hi1))


def const(y):
    """A constant value of y."""

    return Scale(ConstantShape(), y)


def line(y0, y1):
    """The shape of a linear progression from y0 to y1."""

    if y0 < 0:
        raise ValueError('y0 must not be negative, was {}'.format(y0))
    if y1 < 0:
        raise ValueError('y1 must not be negative, was {}'.format(y1))
    return Shift(Scale(LineShape(), y1-y0), y0)


def triangle(y0, y1):
    """A symmetrical triangle from y0 to y1 and back."""

    return Chain(line(y0, y1), line(y1, y0))


def clr(lr_factor=10):
    """A symmetrical triangle from 1/lr_factor to 1 to 1/lr_factor."""

    return triangle(1/lr_factor, 1)


def stlr(lr_factor=10, up_share=1/4):
    """Like the triangle from clr, but asymmetrical if up_share is not 1/2."""

    up = line(1/lr_factor, 1)
    down = line(1, 1/lr_factor)
    return Chain(up, down, first_share=up_share)


def one_cycle(lr_factor=10, anneal_share=1/10, anneal_factor=100):
    """The triangle from clr, plus additional annealing to a tiny factor."""

    core = clr(lr_factor)
    finish = line(1/lr_factor, 1/lr_factor/anneal_factor)
    return Chain(core, finish, first_share=1-anneal_share)


def one_cycle_momentum(y0, y1, anneal_share=1/10):
    """A linear shape from y0 to y1, back to y0, and then flat to anneal."""

    core = triangle(y0, y1)
    finish = const(y0)
    return Chain(core, finish, first_share=1-anneal_share)
