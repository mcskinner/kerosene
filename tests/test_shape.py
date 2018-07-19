import pytest

from kerosene import shape


def test_constant_shape():
    constant = shape.ConstantShape()
    assert constant(0) == 1
    assert constant(1/2) == 1
    assert constant(1) == 1
    assert constant.range() == (1, 1)


def test_line_shape():
    line = shape.LineShape()
    assert line(0) == 0
    assert line(1/2) == 1/2
    assert line(1) == 1
    assert line.range() == (0, 1)


def test_shift():
    line = shape.LineShape()
    shifted = shape.Shift(line, 3)
    assert shifted(0) == 3
    assert shifted(1/2) == 3+1/2
    assert shifted(1) == 4
    assert shifted.range() == (3, 4)


def test_scale():
    line = shape.LineShape()
    scaled = shape.Scale(line, 2)
    assert scaled(0) == 0
    assert scaled(1/2) == 1
    assert scaled(1) == 2
    assert scaled.range() == (0, 2)


def test_two_lines():
    line = shape.LineShape()
    two_lines = shape.Chain(line, line)
    assert two_lines(0) == 0
    assert two_lines(1/4) == 1/2
    assert two_lines(1/2) == 0
    assert two_lines(3/4) == 1/2
    assert two_lines(1) == 1
    assert two_lines.range() == (0, 1)


def test_line():
    line = shape.line(1, 3)
    assert line(0) == 1
    assert line(1/2) == 2
    assert line(1) == 3
    assert line.range() == (1, 3)


def test_clr():
    clr = shape.clr()
    assert clr(0) == pytest.approx(1/10)
    assert clr(1/4) == pytest.approx(5.5/10)
    assert clr(1/2) == pytest.approx(1)
    assert clr(3/4) == pytest.approx(5.5/10)
    assert clr(1) == pytest.approx(1/10)
    lo, hi = clr.range()
    assert lo == pytest.approx(1/10)
    assert hi == pytest.approx(1)


def test_stlr():
    stlr = shape.stlr(20)
    assert stlr(0) == pytest.approx(1/20)
    assert stlr(1/4) == pytest.approx(1)
    assert stlr(5/8) == pytest.approx(10.5/20)
    assert stlr(1) == pytest.approx(1/20)
    lo, hi = stlr.range()
    assert lo == pytest.approx(1/20)
    assert hi == pytest.approx(1)


def test_one_cycle():
    one_cycle = shape.one_cycle()
    assert one_cycle(0) == pytest.approx(1/10)
    assert one_cycle(9/40) == pytest.approx(5.5/10)
    assert one_cycle(18/40) == pytest.approx(1)
    assert one_cycle(27/40) == pytest.approx(5.5/10)
    assert one_cycle(36/40) == pytest.approx(1/10)
    assert one_cycle(38/40) == pytest.approx(50.5/1000)
    assert one_cycle(1) == pytest.approx(1/1000)
    lo, hi = one_cycle.range()
    assert lo == pytest.approx(1/1000)
    assert hi == pytest.approx(1)


def test_one_cycle_momentum():
    momentum = shape.one_cycle_momentum(0.95, 0.85)
    assert momentum(0) == pytest.approx(0.95)
    assert momentum(9/40) == pytest.approx(0.9)
    assert momentum(18/40) == pytest.approx(0.85)
    assert momentum(27/40) == pytest.approx(0.9)
    assert momentum(36/40) == pytest.approx(0.95)
    assert momentum(38/40) == pytest.approx(0.95)
    assert momentum(1) == pytest.approx(0.95)
    lo, hi = momentum.range()
    assert lo == pytest.approx(0.85)
    assert hi == pytest.approx(0.95)
