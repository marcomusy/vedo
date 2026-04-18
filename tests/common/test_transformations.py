from __future__ import annotations

import numpy as np
import pytest

import vedo
from vedo import LinearTransform, NonLinearTransform, Sphere


# ── LinearTransform ──────────────────────────────────────────────────────────

def test_linear_transform_identity() -> None:
    lt = LinearTransform()
    assert lt.is_identity()
    pt = [1.0, 2.0, 3.0]
    assert np.allclose(lt.transform_point(pt), pt)


def test_linear_transform_translate() -> None:
    lt = LinearTransform().translate([10, 20, 30])
    result = lt.transform_point([1, 2, 3])
    assert np.allclose(result, [11, 22, 33])


def test_linear_transform_rotate_z() -> None:
    lt = LinearTransform().rotate_z(90)
    result = lt.transform_point([1, 0, 0])
    assert np.allclose(result, [0, 1, 0], atol=1e-6)


def test_linear_transform_rotate_x() -> None:
    lt = LinearTransform().rotate_x(90)
    result = lt.transform_point([0, 1, 0])
    assert np.allclose(result, [0, 0, 1], atol=1e-6)


def test_linear_transform_rotate_y() -> None:
    lt = LinearTransform().rotate_y(90)
    result = lt.transform_point([0, 0, 1])
    assert np.allclose(result, [1, 0, 0], atol=1e-6)


def test_linear_transform_scale() -> None:
    lt = LinearTransform().scale([2, 3, 4])
    result = lt.transform_point([1, 1, 1])
    assert np.allclose(result, [2, 3, 4], atol=1e-6)


def test_linear_transform_inverse() -> None:
    lt = LinearTransform().translate([5, 6, 7]).rotate_z(45)
    inv = lt.compute_inverse()
    pt = [1.0, 2.0, 0.0]
    roundtrip = inv.transform_point(lt.transform_point(pt))
    assert np.allclose(roundtrip, pt, atol=1e-6)


def test_linear_transform_concatenate() -> None:
    t1 = LinearTransform().translate([1, 0, 0])
    t2 = LinearTransform().translate([0, 2, 0])
    t1.concatenate(t2)
    result = t1.transform_point([0, 0, 0])
    assert np.allclose(result, [1, 2, 0], atol=1e-6)


def test_linear_transform_position_property() -> None:
    lt = LinearTransform().translate([3, 4, 5])
    assert np.allclose(lt.position, [3, 4, 5], atol=1e-6)


def test_linear_transform_matrix_roundtrip() -> None:
    lt = LinearTransform().rotate_z(30).translate([1, 2, 3])
    M = lt.matrix
    lt2 = LinearTransform(M)
    pt = [1.0, 0.0, 0.0]
    assert np.allclose(lt.transform_point(pt), lt2.transform_point(pt), atol=1e-6)


def test_linear_transform_applied_to_mesh() -> None:
    sp = Sphere(r=1)
    lt = LinearTransform().translate([5, 0, 0])
    sp.apply_transform(lt)
    assert np.allclose(sp.center_of_mass(), [5, 0, 0], atol=0.05)


def test_linear_transform_position_after_apply() -> None:
    sp = Sphere(r=1)
    sp.pos(3, 4, 5)
    assert np.allclose(sp.transform.position, [3, 4, 5], atol=1e-4)


def test_linear_transform_rad_mode() -> None:
    lt_deg = LinearTransform().rotate_z(90)
    lt_rad = LinearTransform().rotate_z(np.pi / 2, rad=True)
    pt = [1.0, 0.0, 0.0]
    assert np.allclose(lt_deg.transform_point(pt), lt_rad.transform_point(pt), atol=1e-6)


# ── NonLinearTransform ───────────────────────────────────────────────────────

def test_non_linear_transform_basic() -> None:
    src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tgt = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    nlt = NonLinearTransform()
    nlt.source_points = src
    nlt.target_points = tgt
    nlt.update()
    result = nlt.transform_point([1, 0, 0])
    assert np.allclose(result, [2, 0, 0], atol=0.1)


def test_non_linear_transform_position_always_zero() -> None:
    nlt = NonLinearTransform()
    assert np.allclose(nlt.position, [0, 0, 0])
