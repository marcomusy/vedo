#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pytest

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkImageDataToExplicitStructuredGrid

import vedo
from vedo import utils
from vedo import vtkclasses as vtki


def make_grid() -> vedo.ExplicitStructuredGrid:
    img = vtkImageData()
    img.SetDimensions(2, 2, 2)
    conv = vtkImageDataToExplicitStructuredGrid()
    conv.SetInputData(img)
    conv.Update()
    return vedo.ExplicitStructuredGrid(conv.GetOutput())


def test_explicit_grid_from_points_and_cells() -> None:
    cube = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ])
    points = np.concatenate((cube, cube + (2, 0, 1)))
    cells = np.arange(16).reshape(2, 8)
    grid = vedo.ExplicitStructuredGrid(points, cells=cells, dims=(3, 2, 2))

    assert grid.number_of_cells() == 2
    right = grid.compute_cellid((1, 0, 0))
    np.testing.assert_array_equal(grid.cell_points(right), cells[1])
    np.testing.assert_array_equal(grid.cell_bounds(right), [2, 3, 0, 1, 1, 2])
    assert grid.actor.GetMapper().GetInput().GetNumberOfCells() == 12

    with pytest.raises(ValueError, match="cells must be"):
        vedo.ExplicitStructuredGrid(points, cells=cells[:1], dims=(3, 2, 2))
    cells[1, -1] = len(points)
    with pytest.raises(ValueError, match="point IDs"):
        vedo.ExplicitStructuredGrid(points, cells=cells, dims=(3, 2, 2))


def test_explicit_grid_cellcolors_update_existing_actor() -> None:
    grid = make_grid()
    mapper = grid.actor.GetMapper()
    grid.cellcolors = [[200, 100, 10]]

    expected = np.array([[200, 100, 10, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(grid.cellcolors, expected)
    mapper.Update()
    surface_colors = np.repeat(expected, 6, axis=0)
    np.testing.assert_array_equal(utils.vtk2numpy(mapper.MapScalars(1.0)), surface_colors)

    clone = grid.clone()
    clone_mapper = clone.actor.GetMapper()
    np.testing.assert_array_equal(utils.vtk2numpy(clone_mapper.MapScalars(1.0)), surface_colors)


def test_explicit_grid_basics() -> None:
    empty = vedo.ExplicitStructuredGrid()
    empty.set_dimensions(2, 3, 4)
    assert np.allclose(empty.dimensions(), [2, 3, 4])
    empty.set_extent((1, 2, 3, 5, 7, 10))
    assert np.allclose(empty.extent(), [1, 2, 3, 5, 7, 10])

    grid = make_grid()
    assert np.allclose(grid.dimensions(), [2, 2, 2])
    assert np.allclose(grid.cell_dimensions(), [1, 1, 1])
    assert np.allclose(grid.extent(), [0, 1, 0, 1, 0, 1])
    assert grid.data_dimension() == 3
    assert grid.extent_type() == 1

    assert grid.number_of_cells() == 1
    assert grid.compute_cellid((0, 0, 0)) == 0
    assert np.allclose(grid.compute_cell_structured_coords(0), [0, 0, 0])
    assert np.allclose(grid.cell_points(0), [0, 1, 3, 2, 4, 5, 7, 6])
    assert np.allclose(grid.point_cells(0), [0])
    assert np.allclose(grid.cell_neighbors(0), [-1, -1, -1, -1, -1, -1])
    assert grid.cell_neighbors(0, [0, 1, 3, 2]).size == 0
    assert np.allclose(grid.cell_bounds(0), [0, 1, 0, 1, 0, 1])
    assert grid.cell_type(0) == vtki.cell_types["HEXAHEDRON"]
    assert grid.cell_size(0) == 8
    assert grid.max_cell_size() == 8
    if hasattr(grid.dataset, "GetMaxSpatialDimension"):
        assert grid.max_spatial_dimension() == 3
    if hasattr(grid.dataset, "GetMinSpatialDimension"):
        assert grid.min_spatial_dimension() == 3
    assert grid.find_point((0, 0, 0)) == 0

    assert not grid.has_blank_cells()
    assert not grid.has_blank_points()
    assert not grid.has_ghost_cells()
    assert not grid.has_ghost_points()
    assert grid.is_cell_visible(0)
    assert not grid.is_cell_ghost(0)

    grid.blank_cell(0)
    assert grid.has_blank_cells()
    assert not grid.is_cell_visible(0)
    grid.unblank_cell(0)
    assert grid.is_cell_visible(0)

    assert grid.build_links() is grid
    assert grid.compute_faces_connectivity_flags_array() is grid
    assert grid.check_and_reorder_faces() is grid

    clone = grid.clone()
    assert np.allclose(clone.dimensions(), [2, 2, 2])
