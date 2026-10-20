from __future__ import annotations

import numpy as np
import pytest

from vedo import Line, ThickTube


@pytest.mark.parametrize("resolution", [6, 12, 24])
@pytest.mark.parametrize("as_line", [False, True])
@pytest.mark.parametrize("direction", [(0, 0, 1), (1, 2, 3)])
def test_thick_tube_is_closed_and_oriented(resolution, as_line, direction) -> None:
    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    start = np.array([0.25, -0.5, 1.25])
    points = np.array([start, start + 2 * axis])
    mesh = ThickTube(Line(points) if as_line else points, 0.2, 0.3, res=resolution)

    assert mesh.is_closed()
    assert mesh.is_manifold()
    # Exact volume of two concentric regular polygons extruded by length 2.
    expected = resolution * np.sin(2 * np.pi / resolution) * (0.3**2 - 0.2**2)
    assert mesh.volume() == pytest.approx(expected, rel=2e-5)

    vertices = mesh.coordinates
    faces = np.asarray(mesh.cells)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert mesh.npoints - len(unique_edges) + mesh.ncells == 0

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    np.testing.assert_allclose(mesh.cell_normals, normals, atol=2e-6)
    centers = triangles.mean(axis=1) - start
    axial = centers @ axis
    radial = centers - axial[:, None] * axis
    side = (axial > 1e-5) & (axial < 2 - 1e-5)
    inner = side & (np.linalg.norm(radial, axis=1) < 0.25)
    outer = side & ~inner
    assert np.count_nonzero(inner) == np.count_nonzero(outer) == 2 * resolution
    assert np.all(np.sum(normals[inner] * radial[inner], axis=1) < 0)
    assert np.all(np.sum(normals[outer] * radial[outer], axis=1) > 0)
    assert np.all(normals[axial < 1e-5] @ axis < -0.99)
    assert np.all(normals[axial > 2 - 1e-5] @ axis > 0.99)

    mesh.shift([1, -2, 0.5]).rotate_z(30).scale(2)
    assert mesh.is_closed()
    assert mesh.volume() == pytest.approx(8 * expected, rel=2e-5)
