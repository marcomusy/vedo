from __future__ import annotations

import numpy as np
import pytest

from vedo import Points
from vedo.pointcloud.fits import fit_circle, fit_line, fit_plane, fit_sphere


# ── fit_line ─────────────────────────────────────────────────────────────────

def test_fit_line_pure_x() -> None:
    pts = np.column_stack([np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)])
    line = fit_line(pts)
    assert np.allclose(np.abs(line.slope), [1, 0, 0], atol=0.01)
    assert np.allclose(line.center, [5, 0, 0], atol=0.01)


def test_fit_line_diagonal() -> None:
    t = np.linspace(0, 1, 100)
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    pts = np.outer(t, direction)
    line = fit_line(pts)
    assert np.allclose(np.abs(line.slope), np.abs(direction), atol=0.01)


def test_fit_line_accepts_points_object() -> None:
    pts = np.column_stack([np.linspace(0, 5, 30), np.zeros(30), np.zeros(30)])
    p = Points(pts)
    line = fit_line(p)
    assert np.allclose(np.abs(line.slope), [1, 0, 0], atol=0.01)


def test_fit_line_noisy() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 200)
    pts = np.column_stack([t, rng.normal(0, 0.05, 200), rng.normal(0, 0.05, 200)])
    line = fit_line(pts)
    assert np.allclose(np.abs(line.slope), [1, 0, 0], atol=0.05)


# ── fit_plane ─────────────────────────────────────────────────────────────────

def test_fit_plane_xy() -> None:
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100), np.zeros(100)])
    plane = fit_plane(pts)
    assert np.allclose(np.abs(plane.normal), [0, 0, 1], atol=0.01)
    assert np.allclose(plane.center[2], 0, atol=0.01)


def test_fit_plane_xz() -> None:
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-1, 1, 100), np.zeros(100), rng.uniform(-1, 1, 100)])
    plane = fit_plane(pts)
    assert np.allclose(np.abs(plane.normal), [0, 1, 0], atol=0.01)


def test_fit_plane_offset() -> None:
    rng = np.random.default_rng(3)
    z0 = 5.0
    pts = np.column_stack([rng.uniform(-1, 1, 80), rng.uniform(-1, 1, 80), np.full(80, z0)])
    plane = fit_plane(pts)
    assert np.allclose(np.abs(plane.normal), [0, 0, 1], atol=0.01)
    assert np.allclose(plane.center[2], z0, atol=0.01)


def test_fit_plane_noisy() -> None:
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(-2, 2, 300), rng.normal(0, 0.02, 300)])
    plane = fit_plane(pts)
    assert np.allclose(np.abs(plane.normal), [0, 0, 1], atol=0.05)


# ── fit_sphere ────────────────────────────────────────────────────────────────

def test_fit_sphere_unit() -> None:
    theta = np.linspace(0, 2 * np.pi, 30)
    phi = np.linspace(0, np.pi, 15)
    T, P = np.meshgrid(theta, phi)
    pts = np.column_stack([
        np.sin(P.ravel()) * np.cos(T.ravel()),
        np.sin(P.ravel()) * np.sin(T.ravel()),
        np.cos(P.ravel()),
    ])
    sph = fit_sphere(pts)
    assert sph is not None
    assert np.isclose(sph.radius, 1.0, atol=0.01)
    assert np.allclose(sph.center, [0, 0, 0], atol=0.01)


def test_fit_sphere_offset_center() -> None:
    center = np.array([3.0, -1.0, 2.0])
    r = 2.5
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 10)
    T, P = np.meshgrid(theta, phi)
    pts = center + r * np.column_stack([
        np.sin(P.ravel()) * np.cos(T.ravel()),
        np.sin(P.ravel()) * np.sin(T.ravel()),
        np.cos(P.ravel()),
    ])
    sph = fit_sphere(pts)
    assert sph is not None
    assert np.isclose(sph.radius, r, atol=0.05)
    assert np.allclose(sph.center, center, atol=0.05)


def test_fit_sphere_noisy() -> None:
    rng = np.random.default_rng(5)
    r = 3.0
    n = 500
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * r
    pts += rng.normal(0, 0.02, pts.shape)
    sph = fit_sphere(pts)
    assert sph is not None
    assert np.isclose(sph.radius, r, atol=0.1)


# ── fit_circle ────────────────────────────────────────────────────────────────

def test_fit_circle_xy_plane() -> None:
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    r = 4.0
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(60)])
    center, radius, normal = fit_circle(pts)
    assert np.isclose(radius, r, atol=0.05)
    assert np.allclose(np.abs(normal), [0, 0, 1], atol=0.01)


def test_fit_circle_offset_center() -> None:
    cx, cy = 2.0, -3.0
    r = 1.5
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles), np.zeros(50)])
    center, radius, normal = fit_circle(pts)
    assert np.isclose(radius, r, atol=0.05)
    assert np.allclose(center[:2], [cx, cy], atol=0.05)
