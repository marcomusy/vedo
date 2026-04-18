from __future__ import annotations

import numpy as np
import pytest

import vedo
from vedo.addons import SphereWidget


def test_sphere_widget_construction() -> None:
    sw = SphereWidget(center=(1, 2, 3), r=2.0)
    assert np.allclose(sw.center, [1, 2, 3], atol=1e-5)
    assert np.isclose(sw.radius, 2.0, atol=1e-5)


def test_sphere_widget_defaults() -> None:
    sw = SphereWidget()
    assert np.isclose(sw.radius, 1.0, atol=1e-5)
    assert np.allclose(sw.center, [0, 0, 0], atol=1e-5)


def test_sphere_widget_setters() -> None:
    sw = SphereWidget()
    sw.center = [3, 4, 5]
    sw.radius = 7.0
    assert np.allclose(sw.center, [3, 4, 5], atol=1e-5)
    assert np.isclose(sw.radius, 7.0, atol=1e-5)


def test_sphere_widget_chaining() -> None:
    sw = SphereWidget()
    result = sw.color("blue").alpha(0.3).lw(2)
    assert result is sw


def test_sphere_widget_get_sphere() -> None:
    sw = SphereWidget(center=(1, 0, 0), r=1.5)
    s = sw.get_sphere()
    assert isinstance(s, vedo.shapes.Sphere)


def test_sphere_widget_is_enabled_before_render() -> None:
    sw = SphereWidget()
    assert not sw.is_enabled()



def test_sphere_widget_top_level_import() -> None:
    sw = vedo.SphereWidget(center=(0, 0, 0), r=2)
    assert isinstance(sw, SphereWidget)


def test_sphere_widget_points() -> None:
    sw = SphereWidget(center=(1, 2, 3), r=2.0, res=12)
    pts = sw.points
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] > 0
    assert np.allclose(pts.mean(axis=0), [1, 2, 3], atol=0.1)


def test_sphere_widget_repr() -> None:
    sw = SphereWidget(center=(1, 2, 3), r=4)
    r = repr(sw)
    assert "SphereWidget" in r
    assert "center" in r
    assert "radius" in r
