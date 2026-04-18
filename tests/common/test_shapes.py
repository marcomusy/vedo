from __future__ import annotations

import numpy as np
import pytest

import vedo
from vedo.shapes import (
    Arc,
    Arrow,
    Box,
    Cone,
    Cube,
    Cylinder,
    Line,
    Sphere,
    Spline,
    Text3D,
    Torus,
    Tube,
)
from vedo.pointcloud.fits import pca_ellipsoid


def test_arc_construction() -> None:
    arc = Arc(center=None, point1=(1, 1, 1), point2=None, normal=(0, 0, 1), angle=np.pi)
    assert isinstance(arc, Arc)


def test_sphere() -> None:
    sp = Sphere(r=2, res=12)
    assert sp.npoints > 0
    assert sp.ncells > 0
    assert np.allclose(sp.center_of_mass(), [0, 0, 0], atol=0.01)
    b = sp.bounds()
    assert np.allclose(b[0], -2, atol=0.05)
    assert np.allclose(b[1],  2, atol=0.05)


def test_box() -> None:
    b = Box(length=2, width=3, height=4)
    assert b.npoints > 0
    assert np.allclose(b.xbounds(), [-1, 1], atol=0.01)
    assert np.allclose(b.ybounds(), [-1.5, 1.5], atol=0.01)
    assert np.allclose(b.zbounds(), [-2, 2], atol=0.01)


def test_cube() -> None:
    c = Cube(side=3)
    assert c.npoints > 0
    assert np.allclose(c.diagonal_size(), 3 * np.sqrt(3), atol=0.1)


def test_cylinder() -> None:
    cyl = Cylinder(pos=(0, 0, 0), r=1, height=4, res=16)
    assert cyl.npoints > 0
    zb = cyl.zbounds()
    assert np.allclose(zb, [-2, 2], atol=0.05)
    assert np.allclose(cyl.center_of_mass(), [0, 0, 0], atol=0.05)


def test_cone() -> None:
    cn = Cone(pos=(0, 0, 0), r=1, height=3, res=16)
    assert cn.npoints > 0
    zb = cn.zbounds()
    assert zb[0] < 0 < zb[1]


def test_torus() -> None:
    t = Torus(r1=3, r2=0.5, res=16)
    assert t.npoints > 0
    assert np.allclose(t.center_of_mass(), [0, 0, 0], atol=0.1)
    xb = t.xbounds()
    assert xb[0] < -3
    assert xb[1] > 3


def test_ellipsoid_asphericity_error_fix() -> None:
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((200, 3)) * [1, 2, 3]
    ell = pca_ellipsoid(pts)
    assert ell is not None
    err = ell.asphericity_error()
    assert isinstance(err, float)
    assert err >= 0
    # va_error, vb_error, vc_error should all be positive and independent
    assert ell.va_error > 0
    assert ell.vb_error > 0
    assert ell.vc_error > 0
    assert not np.isclose(ell.vb_error, ell.vc_error), (
        "vb_error == vc_error — the ec=b bug may have regressed"
    )


def test_line() -> None:
    ln = Line([0, 0, 0], [1, 2, 3])
    assert ln.npoints == 2
    assert np.allclose(ln.vertices[0], [0, 0, 0])
    assert np.allclose(ln.vertices[-1], [1, 2, 3])


def test_tube() -> None:
    pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    tb = Tube(pts, r=0.1)
    assert tb.npoints > 0
    assert tb.ncells > 0


def test_spline() -> None:
    pts = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]]
    sp = Spline(pts, res=20)
    assert sp.npoints >= 20


def test_arrow() -> None:
    ar = Arrow([0, 0, 0], [1, 0, 0])
    assert ar.npoints > 0
    xb = ar.xbounds()
    assert xb[0] >= -0.01
    assert xb[1] > 0.9


def test_text3d() -> None:
    tx = Text3D("Hello", pos=(1, 2, 3), s=0.5)
    assert tx.npoints > 0
    # anchor is at pos
    b = tx.bounds()
    assert b[0] >= 0.9  # xmin close to x=1
    assert tx.txt == "Hello"


def test_text3d_justify() -> None:
    tx = Text3D("Hi", pos=(0, 0, 0), s=1, justify="centered")
    com = tx.center_of_mass()
    assert np.allclose(com[:2], [0, 0], atol=0.5)


def test_shape_color_and_alpha() -> None:
    sp = Sphere()
    sp.color("red")
    c = sp.color()
    assert c[0] > 0.8  # strong red component
    sp.alpha(0.3)
    assert np.isclose(sp.alpha(), 0.3)


def test_shape_pos_and_shift() -> None:
    sp = Sphere()
    sp.pos(5, 6, 7)
    assert np.allclose(sp.pos(), [5, 6, 7], atol=0.01)
    sp.shift(1, 0, 0)
    assert np.allclose(sp.pos(), [6, 6, 7], atol=0.01)
    sp.shift([0, 1, 0])
    assert np.allclose(sp.pos(), [6, 7, 7], atol=0.01)
