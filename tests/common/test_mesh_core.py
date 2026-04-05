from __future__ import annotations

import numpy as np
import vtk

from vedo import Assembly, Cone, Sphere, Volume, merge, transformations


def test_surface_and_volume_regressions() -> None:
    cone = Cone(res=48)
    sphere = Sphere(res=24)

    cone.pointdata["parr"] = cone.vertices[:, 0]
    cone.celldata["carr"] = cone.cell_centers().coordinates[:, 2]

    sphere.pointdata["parr"] = sphere.vertices[:, 0]
    sphere.celldata["carr"] = sphere.cell_centers().coordinates[:, 2]
    sphere.pointdata["pvectors"] = np.sin(sphere.vertices)

    sphere.compute_elevation()
    cone.compute_normals()
    sphere.compute_normals()

    clone = cone.clone()
    assert cone.npoints == clone.npoints
    assert cone.ncells == clone.ncells

    merged = merge(sphere, cone)
    assert merged.npoints == cone.npoints + sphere.npoints
    assert merged.ncells == cone.ncells + sphere.ncells

    assert isinstance(cone.dataset, vtk.vtkPolyData)
    assert isinstance(cone.mapper, vtk.vtkPolyDataMapper)

    cone.pickable(False)
    cone.pickable(True)
    assert cone.pickable()

    cone.pos(1, 2, 3)
    assert np.allclose([1, 2, 3], cone.pos())
    cone.pos(5, 6)
    assert np.allclose([5, 6, 0], cone.pos())

    cone.pos(5, 6, 7).shift(3, 0, 0)
    assert np.allclose([8, 6, 7], cone.pos(), atol=0.001)

    cone.pos(10, 11, 12)
    cone.x(1.1)
    assert np.allclose([1.1, 11, 12], cone.pos(), atol=0.001)
    cone.y(1.2)
    assert np.allclose([1.1, 1.2, 12], cone.pos(), atol=0.001)
    cone.z(1.3)
    assert np.allclose([1.1, 1.2, 1.3], cone.pos(), atol=0.001)

    reoriented = cone.pos(0, 0, 0).clone().rotate(90, axis=(0, 1, 0))
    assert np.max(reoriented.vertices[:, 2]) < 1.01

    reoriented = cone.pos(0, 0, 0).clone().reorient([0, 0, 1], (1, 1, 0))
    assert np.max(reoriented.vertices[:, 2]) < 1.01

    reoriented.scale(5)
    assert np.max(reoriented.vertices[:, 2]) > 4.98

    bounds_box = cone.box()
    assert bounds_box.npoints == 24
    assert bounds_box.clean().npoints == 8

    transformed = cone.clone().rotate_x(10).rotate_y(10).rotate_z(10)
    assert isinstance(transformed.transform.T, vtk.vtkTransform)
    assert transformed.transform.T.GetNumberOfConcatenatedTransforms()

    assert "parr" in cone.pointdata.keys()
    assert "carr" in cone.celldata.keys()

    point_array = sphere.pointdata["parr"]
    assert len(point_array)
    assert np.max(point_array) > 0.99

    cell_array = sphere.celldata["carr"]
    assert len(cell_array)
    assert np.max(cell_array) > 0.99

    assert isinstance(cone + sphere, Assembly)

    shifted = sphere.clone()
    shifted.vertices = sphere.vertices + [1, 2, 3]
    assert np.allclose(shifted.vertices, sphere.vertices + [1, 2, 3], atol=0.001)

    assert np.array(sphere.cells).shape == (2112, 3)

    textured = sphere.clone().texture(np.full((8, 8, 3), 180, dtype=np.uint8))
    assert isinstance(textured.actor.GetTexture(), vtk.vtkTexture)

    trimmed = sphere.clone().delete_cells_by_point_index(range(100))
    assert trimmed.npoints == sphere.npoints
    assert trimmed.ncells < sphere.ncells

    quantized = sphere.clone().quantize(0.1)
    assert quantized.npoints == 834

    scaled_sphere = sphere.clone().scale([1, 2, 3])
    assert np.allclose(scaled_sphere.xbounds(), [-1, 1], atol=0.01)
    assert np.allclose(scaled_sphere.ybounds(), [-2, 2], atol=0.01)
    assert np.allclose(scaled_sphere.zbounds(), [-3, 3], atol=0.01)

    assert 9.9 < Sphere().scale(10).pos(1, 3, 7).average_size() < 10.1
    assert 3.3 < sphere.diagonal_size() < 3.5
    assert np.allclose(sphere.center_of_mass(), [0, 0, 0], atol=0.001)
    assert 4.1 < sphere.volume() < 4.2
    assert 12.5 < sphere.area() < 12.6

    point = [12, 34, 52]
    assert np.allclose(
        sphere.closest_point(point),
        [0.19883616, 0.48003298, 0.85441941],
        atol=0.001,
    )

    in_bounds = sphere.find_cells_in_bounds(xbounds=(-0.5, 0.5))
    assert len(in_bounds) == 1576

    transformed_sphere = Sphere().apply_transform(cone.clone().pos(35, 67, 87).transform)
    assert np.allclose(transformed_sphere.center_of_mass(), (35, 67, 87), atol=0.001)

    normalized = sphere.clone().pos(10, 20, 30).scale([7, 8, 9]).normalize()
    assert np.allclose(normalized.center_of_mass(), (10, 20, 30), atol=0.001)
    assert 0.9 < normalized.average_size() < 1.1

    cropped = cone.clone().crop(left=0.5)
    assert np.min(cropped.vertices[:, 0]) > -0.001

    subdivided = sphere.clone().subdivide(4)
    assert subdivided.npoints == 270338

    decimated = sphere.clone().decimate(0.2)
    assert decimated.npoints == 213

    assert np.allclose(
        sphere.vertex_normals[12],
        [9.97668684e-01, 1.01513637e-04, 6.82437494e-02],
        atol=0.001,
    )
    assert Sphere().contains([0.1, 0.2, 0.3])

    assembly = cone + sphere
    assert len(assembly.unpack()) == 2
    assert assembly.unpack(0) == cone
    assert assembly.unpack(1) == sphere
    assert 4.1 < assembly.diagonal_size() < 4.2

    x, y, z = np.mgrid[:30, :30, :30]
    scalar_field = ((x - 15) ** 2 + (y - 15) ** 2 + (z - 15) ** 2) / 225
    volume = Volume(scalar_field)
    volume_array = volume.pointdata[0]

    assert volume_array.shape[0] == 27000
    assert np.max(volume_array) == 3
    assert np.min(volume_array) == 0

    iso = volume.isosurface(1.0)
    assert 2540 < iso.area() < 3000

    coords = [5, 2, 3]
    coords = transformations.cart2spher(*coords)
    coords = transformations.spher2cart(*coords)
    assert np.allclose(coords, [5, 2, 3], atol=0.001)
    coords = transformations.cart2cyl(*coords)
    coords = transformations.cyl2cart(*coords)
    assert np.allclose(coords, [5, 2, 3], atol=0.001)
    coords = transformations.cart2cyl(*coords)
    coords = transformations.cyl2spher(*coords)
    coords = transformations.spher2cart(*coords)
    assert np.allclose(coords, [5, 2, 3], atol=0.001)
    coords = transformations.cart2spher(*coords)
    coords = transformations.spher2cyl(*coords)
    coords = transformations.cyl2cart(*coords)
    assert np.allclose(coords, [5, 2, 3], atol=0.001)
