from __future__ import annotations

import numpy as np
import vedo
from vedo.addons import CylinderWidget

BOUNDS = [-5, 5, -5, 5, -5, 5]


def test_cylinder_widget_construction() -> None:
    cw = CylinderWidget((5, 0, 30), [-1,17,-6,6,23,37], r=2, axis=(0, 0, 1))
    assert np.allclose(cw.center, [5, 0, 30], atol=1e-5)
    assert np.isclose(cw.radius, 2.0, atol=1e-5)
    assert np.allclose(cw.axis, [0, 0, 1], atol=1e-5)


def test_cylinder_widget_setters() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS)
    cw.center = [1, 2, 3]
    cw.radius = 0.5
    cw.axis = [0, 1, 0]
    assert np.allclose(cw.center, [1, 2, 3], atol=1e-5)
    assert np.isclose(cw.radius, 0.5, atol=1e-5)
    assert np.allclose(cw.axis, [0, 1, 0], atol=1e-5)


def test_cylinder_widget_axis_normalized() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS, axis=(0, 0, 2))
    assert np.isclose(np.linalg.norm(cw.axis), 1.0, atol=1e-5)


def test_cylinder_widget_points_shape() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS, r=1, res=(8, 5))
    pts = cw.points
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] == 8 * 5  # radial × axial


def test_cylinder_widget_points_shape_scalar_res() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS, r=1, res=6)
    pts = cw.points
    assert pts.shape[0] == 6 * 6  # both dims = 6


def test_cylinder_widget_points_center() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS, r=1, axis=(0, 0, 1), res=16)
    pts = cw.points
    assert np.allclose(pts.mean(axis=0), [0, 0, 0], atol=0.1)


def test_cylinder_widget_bounds_from_vedo_object() -> None:
    sphere = vedo.Sphere()
    cw = CylinderWidget((0, 0, 0), sphere, r=0.5)
    assert np.allclose(cw.center, [0, 0, 0], atol=1e-5)


def test_cylinder_widget_chaining() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS)
    result = cw.color("blue").alpha(0.4)
    assert result is cw


def test_cylinder_widget_is_enabled() -> None:
    cw = CylinderWidget((0, 0, 0), BOUNDS)
    assert not cw.is_enabled()


def test_cylinder_widget_top_level_import() -> None:
    cw = vedo.CylinderWidget((0, 0, 0), BOUNDS)
    assert isinstance(cw, CylinderWidget)


def test_cylinder_widget_repr() -> None:
    cw = CylinderWidget((1, 2, 3), BOUNDS, r=2, axis=(0, 1, 0))
    r = repr(cw)
    assert "CylinderWidget" in r
    assert "center" in r
    assert "radius" in r
