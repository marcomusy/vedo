from __future__ import annotations

import numpy as np
import pytest

import vedo
from vedo.addons import LineWidget


def test_line_widget_construction() -> None:
    lw = LineWidget((-1, 0, 0), (1, 0, 0))
    assert np.allclose(lw.p1, [-1, 0, 0], atol=1e-5)
    assert np.allclose(lw.p2, [1, 0, 0], atol=1e-5)


def test_line_widget_geometry() -> None:
    lw = LineWidget((0, 0, 0), (3, 4, 0))
    assert np.isclose(lw.length, 5.0, atol=1e-5)
    assert np.allclose(lw.midpoint, [1.5, 2.0, 0.0], atol=1e-5)
    assert np.allclose(lw.direction, [0.6, 0.8, 0.0], atol=1e-5)


def test_line_widget_zero_length_direction() -> None:
    lw = LineWidget((1, 2, 3), (1, 2, 3))
    assert np.allclose(lw.direction, [0, 0, 0])


def test_line_widget_setters() -> None:
    lw = LineWidget((0, 0, 0), (1, 0, 0))
    lw.p1 = [5, 6, 7]
    lw.p2 = [8, 9, 10]
    assert np.allclose(lw.p1, [5, 6, 7], atol=1e-5)
    assert np.allclose(lw.p2, [8, 9, 10], atol=1e-5)


def test_line_widget_chaining() -> None:
    lw = LineWidget()
    result = lw.line_color("red").handle_color("blue").lw(3).ps(12)
    assert result is lw


def test_line_widget_get_line() -> None:
    lw = LineWidget((-1, 0, 0), (1, 0, 0))
    line = lw.get_line()
    assert isinstance(line, vedo.Line)


def test_line_widget_defaults() -> None:
    lw = LineWidget()
    assert np.isclose(lw.length, 1.0, atol=0.01)


def test_line_widget_show_distance() -> None:
    lw = LineWidget()
    lw.show_distance(True)
    lw.show_distance(False)


def test_line_widget_toggle_before_render() -> None:
    lw = LineWidget()
    assert not lw.is_enabled()


def test_line_widget_top_level_import() -> None:
    lw = vedo.LineWidget((0, 0, 0), (1, 1, 1))
    assert isinstance(lw, LineWidget)


def test_line_widget_repr() -> None:
    lw = LineWidget((0, 0, 0), (1, 0, 0))
    r = repr(lw)
    assert "LineWidget" in r
    assert "length" in r
