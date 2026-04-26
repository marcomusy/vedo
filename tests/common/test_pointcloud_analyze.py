from __future__ import annotations

import numpy as np
import pytest

import vedo


# ── compute_acoplanarity ──────────────────────────────────────────────────────

def _flat_grid(n=10):
    """Regular n×n grid at z=0."""
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(n * n)])


def _sphere_pts(n=200, seed=0):
    """n random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def test_flat_plane_all_zero():
    cloud = vedo.Points(_flat_grid())
    cloud.compute_acoplanarity(n=9)
    vals = cloud.pointdata["Acoplanarity"]
    assert vals.shape == (100,)
    assert np.all(vals == 0.0)


def test_sphere_positive():
    cloud = vedo.Points(_sphere_pts())
    cloud.compute_acoplanarity(n=15)
    vals = cloud.pointdata["Acoplanarity"]
    assert vals.shape == (200,)
    assert np.all(vals >= 0.0)
    assert vals.mean() > 1e-3


def test_curved_greater_than_flat():
    flat = vedo.Points(_flat_grid())
    flat.compute_acoplanarity(n=9)

    sphere = vedo.Points(_sphere_pts())
    sphere.compute_acoplanarity(n=15)

    assert sphere.pointdata["Acoplanarity"].mean() > flat.pointdata["Acoplanarity"].mean()


def test_radius_mode_flat():
    cloud = vedo.Points(_flat_grid())
    cloud.compute_acoplanarity(n=0, radius=0.25)
    vals = cloud.pointdata["Acoplanarity"]
    assert vals.shape == (100,)
    assert np.all(vals[vals >= 0] == 0.0)


def test_cell_mode_flat():
    msh = vedo.Plane(res=(6, 6))
    msh.compute_acoplanarity(n=6, on="cells")
    vals = msh.celldata["Acoplanarity"]
    assert len(vals) == msh.ncells
    assert np.all(vals == 0.0)


def test_cell_mode_sphere():
    msh = vedo.Sphere(res=8)
    msh.compute_acoplanarity(n=6, on="cells")
    vals = msh.celldata["Acoplanarity"]
    assert len(vals) == msh.ncells
    assert vals.mean() > 1e-3


def test_raises_no_n_no_radius():
    cloud = vedo.Points(_flat_grid())
    with pytest.raises(ValueError):
        cloud.compute_acoplanarity(n=0, radius=None)


def test_raises_invalid_on():
    cloud = vedo.Points(_flat_grid())
    with pytest.raises(ValueError):
        cloud.compute_acoplanarity(n=9, on="edges")


# ── smooth_lloyd_2d ───────────────────────────────────────────────────────────

def _random_cloud_2d(n=60, seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    return vedo.Points(np.column_stack([xy, np.zeros(n)]))


def test_lloyd_preserves_count():
    cloud = _random_cloud_2d()
    result = cloud.smooth_lloyd_2d(iterations=2)
    assert result.npoints == cloud.npoints


def test_lloyd_returns_2d_points():
    """Output z-coordinates should all be zero (2-D operation)."""
    cloud = _random_cloud_2d()
    result = cloud.smooth_lloyd_2d(iterations=2)
    assert np.allclose(result.vertices[:, 2], 0.0)


def test_lloyd_stays_within_bounds():
    cloud = _random_cloud_2d()
    bds = [0.0, 1.0, 0.0, 1.0]
    result = cloud.smooth_lloyd_2d(iterations=5, bounds=bds)
    xy = result.vertices[:, :2]
    assert np.all(xy[:, 0] >= bds[0]) and np.all(xy[:, 0] <= bds[1])
    assert np.all(xy[:, 1] >= bds[2]) and np.all(xy[:, 1] <= bds[3])


def test_lloyd_moves_points():
    """Relaxation should actually change point positions."""
    cloud = _random_cloud_2d()
    result = cloud.smooth_lloyd_2d(iterations=5)
    assert not np.allclose(cloud.vertices[:, :2], result.vertices[:, :2])


def test_lloyd_increases_min_distance():
    """After relaxation points should be more evenly spread: minimum
    pairwise distance should be greater than before relaxation."""
    from scipy.spatial import distance_matrix
    cloud = _random_cloud_2d(n=40)
    before = cloud.vertices[:, :2]
    result = cloud.smooth_lloyd_2d(iterations=10)
    after = result.vertices[:, :2]

    def _min_dist(pts):
        d = distance_matrix(pts, pts)
        np.fill_diagonal(d, np.inf)
        return d.min()

    assert _min_dist(after) >= _min_dist(before)


def test_lloyd_zero_iterations_unchanged():
    cloud = _random_cloud_2d()
    result = cloud.smooth_lloyd_2d(iterations=0)
    assert np.allclose(cloud.vertices[:, :2], result.vertices[:, :2])


def test_lloyd_name_and_pipeline():
    cloud = _random_cloud_2d()
    result = cloud.smooth_lloyd_2d(iterations=1)
    assert result.name == "MeshSmoothLloyd2D"
    assert result.pipeline is not None
