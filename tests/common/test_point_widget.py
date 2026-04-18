from __future__ import annotations

import numpy as np
import vedo
from vedo.addons import PointWidget


def test_point_widget_construction() -> None:
    pw = PointWidget((1, 2, 3))
    assert np.allclose(pw.pos, [1, 2, 3], atol=1e-5)


def test_point_widget_default_pos() -> None:
    pw = PointWidget()
    assert np.allclose(pw.pos, [0, 0, 0], atol=1e-5)


def test_point_widget_setter() -> None:
    pw = PointWidget((0, 0, 0))
    pw.pos = [7, 8, 9]
    assert np.allclose(pw.pos, [7, 8, 9], atol=1e-5)


def test_point_widget_chaining() -> None:
    pw = PointWidget()
    result = pw.color("blue").alpha(0.5).ps(8)
    assert result is pw


def test_point_widget_is_enabled_before_render() -> None:
    pw = PointWidget()
    assert not pw.is_enabled()


def test_point_widget_top_level_import() -> None:
    pw = vedo.PointWidget((1, 0, 0))
    assert isinstance(pw, PointWidget)


def test_point_widget_repr() -> None:
    pw = PointWidget((3, 4, 5))
    r = repr(pw)
    assert "PointWidget" in r
    assert "pos" in r
