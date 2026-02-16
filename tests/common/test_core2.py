from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from vedo import (
    Ellipsoid,
    Line,
    Points,
    RectilinearGrid,
    Sphere,
    TetMesh,
    Volume,
    dataurl,
)


def _make_datasets():
    vol = Volume(dataurl + "embryo.tif")
    tm = TetMesh(dataurl + "limb.vtu")
    rg = RectilinearGrid(dataurl + "RectilinearGrid.vtr")
    return vol, tm, rg


def _make_line():
    return Line([0, 0, 0], [1, 1, 1], res=100)


def test_add_ids():
    print("\n--- test_add_ids ---")
    vol, tm, rg = _make_datasets()
    assert vol.add_ids() is vol
    assert tm.add_ids() is tm
    assert rg.add_ids() is rg


def test_average_size():
    print("\n--- test_average_size ---")
    vol, tm, rg = _make_datasets()
    assert vol.average_size() > 0
    assert tm.average_size() > 0
    assert rg.average_size() > 0


def test_bounds():
    print("\n--- test_bounds ---")
    vol, tm, rg = _make_datasets()
    assert len(vol.bounds()) == 6
    assert len(tm.bounds()) == 6
    assert len(rg.bounds()) == 6


def test_cell_centers():
    print("\n--- test_cell_centers ---")
    vol, tm, rg = _make_datasets()
    assert vol.cell_centers().coordinates.shape[1] == 3
    assert tm.cell_centers().coordinates.shape[1] == 3
    assert rg.cell_centers().coordinates.shape[1] == 3


def test_cells():
    print("\n--- test_cells ---")
    vol, tm, rg = _make_datasets()
    assert len(vol.cells) == 0
    assert len(tm.cells) == tm.ncells
    assert len(rg.cells) == 0


def test_center_of_mass():
    print("\n--- test_center_of_mass ---")
    vol, tm, rg = _make_datasets()
    assert len(vol.center_of_mass()) == 3
    assert len(tm.center_of_mass()) == 3
    assert len(rg.center_of_mass()) == 3


def test_compute_cell_size():
    print("\n--- test_compute_cell_size ---")
    vol, tm, rg = _make_datasets()
    assert vol.compute_cell_size() is vol
    assert tm.compute_cell_size() is tm
    assert rg.compute_cell_size() is rg


def test_copy_data_from():
    print("\n--- test_copy_data_from ---")
    vol, tm, rg = _make_datasets()
    assert vol.clone().copy_data_from(vol) is not None
    assert tm.clone().copy_data_from(tm) is not None
    assert rg.clone().copy_data_from(rg) is not None


def test_find_cells_along_line():
    print("\n--- test_find_cells_along_line ---")
    vol, tm, rg = _make_datasets()
    assert isinstance(vol.find_cells_along_line([0, 0, 0], [1000, 1000, 1000]), np.ndarray)
    assert isinstance(tm.find_cells_along_line([0, 0, 0], [100, 1, 1]), np.ndarray)
    assert isinstance(rg.find_cells_along_line([0, 0, 0], [10, 1, 1]), np.ndarray)


def test_find_cells_in_bounds():
    print("\n--- test_find_cells_in_bounds ---")
    vol, tm, rg = _make_datasets()
    b = Sphere().bounds()
    assert isinstance(vol.find_cells_in_bounds(b), np.ndarray)
    assert isinstance(tm.find_cells_in_bounds(b), np.ndarray)
    assert isinstance(rg.find_cells_in_bounds(b), np.ndarray)


def test_integrate_data():
    print("\n--- test_integrate_data ---")
    vol, tm, rg = _make_datasets()
    assert vol.integrate_data() is not None
    assert tm.integrate_data() is not None
    assert rg.integrate_data() is not None


def test_interpolate_data_from():
    print("\n--- test_interpolate_data_from ---")
    vol, tm, rg = _make_datasets()
    assert vol.interpolate_data_from(vol, n=1) is vol
    assert tm.interpolate_data_from(vol, n=1) is tm
    assert rg.interpolate_data_from(vol, n=1) is rg


def test_map_cells_to_points():
    print("\n--- test_map_cells_to_points ---")
    vol, tm, rg = _make_datasets()
    assert vol.clone().map_cells_to_points() is not None
    assert tm.clone().map_cells_to_points() is not None
    assert rg.clone().map_cells_to_points() is not None


def test_map_points_to_cells():
    print("\n--- test_map_points_to_cells ---")
    vol, tm, rg = _make_datasets()
    assert vol.clone().map_points_to_cells() is not None
    assert tm.clone().map_points_to_cells() is not None
    assert rg.clone().map_points_to_cells() is not None


def test_lines():
    print("\n--- test_lines ---")
    vol, tm, rg = _make_datasets()
    assert isinstance(vol.lines, np.ndarray)
    assert isinstance(tm.lines, np.ndarray)
    assert isinstance(rg.lines, np.ndarray)


def test_lines_as_flat_array():
    print("\n--- test_lines_as_flat_array ---")
    vol, tm, rg = _make_datasets()
    assert isinstance(vol.lines_as_flat_array, np.ndarray)
    assert isinstance(tm.lines_as_flat_array, np.ndarray)
    assert isinstance(rg.lines_as_flat_array, np.ndarray)


def test_mark_boundaries():
    print("\n--- test_mark_boundaries ---")
    vol, tm, rg = _make_datasets()
    assert vol.mark_boundaries() is vol
    assert tm.mark_boundaries() is tm
    assert rg.mark_boundaries() is rg


def test_memory_address():
    print("\n--- test_memory_address ---")
    vol, tm, rg = _make_datasets()
    assert isinstance(vol.memory_address(), int) and vol.memory_address() > 0
    assert isinstance(tm.memory_address(), int) and tm.memory_address() > 0
    assert isinstance(rg.memory_address(), int) and rg.memory_address() > 0


def test_memory_size():
    print("\n--- test_memory_size ---")
    vol, tm, rg = _make_datasets()
    assert vol.memory_size() > 0
    assert tm.memory_size() > 0
    assert rg.memory_size() > 0


def test_modified():
    print("\n--- test_modified ---")
    vol, tm, rg = _make_datasets()
    assert vol.modified() is vol
    assert tm.modified() is tm
    assert rg.modified() is rg


def test_npoints():
    print("\n--- test_npoints ---")
    vol, tm, rg = _make_datasets()
    assert vol.npoints > 0
    assert tm.npoints > 0
    assert rg.npoints > 0


def test_ncells():
    print("\n--- test_ncells ---")
    vol, tm, rg = _make_datasets()
    assert vol.ncells > 0
    assert tm.ncells > 0
    assert rg.ncells > 0


def test_probe():
    print("\n--- test_probe ---")
    vol, tm, rg = _make_datasets()
    line = _make_line()
    assert line.probe(vol) is not None
    assert line.probe(tm) is not None
    assert line.probe(rg) is not None


def test_resample_data_from():
    print("\n--- test_resample_data_from ---")
    vol, tm, rg = _make_datasets()
    assert vol.clone().resample_data_from(vol) is not None
    assert tm.clone().resample_data_from(tm) is not None
    assert rg.clone().resample_data_from(rg) is not None


def test_smooth_data():
    print("\n--- test_smooth_data ---")
    vol, tm, rg = _make_datasets()
    assert vol.smooth_data() is vol
    assert tm.smooth_data() is tm
    assert rg.smooth_data() is rg


def test_shrink():
    print("\n--- test_shrink ---")
    _, tm, _ = _make_datasets()
    shrunk = tm.shrink()
    assert shrunk is not None
    assert shrunk.ncells == tm.ncells


def test_tomesh():
    print("\n--- test_tomesh ---")
    vol, tm, rg = _make_datasets()
    assert vol.tomesh() is not None
    assert tm.tomesh() is not None
    assert rg.tomesh() is not None


def test_write():
    print("\n--- test_write ---")
    vol, tm, rg = _make_datasets()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        vti = out_dir / "test.vti"
        vtu = out_dir / "test.vtu"
        vtr = out_dir / "test.vtr"

        vol.write(str(vti))
        tm.write(str(vtu))
        rg.write(str(vtr))

        assert vti.exists()
        assert vtu.exists()
        assert vtr.exists()


def test_cut_with_mesh():
    print("\n--- test_cut_with_mesh ---")
    _, tm, rg = _make_datasets()
    cut_surface = Ellipsoid().scale(5)
    assert tm.cut_with_mesh(cut_surface) is not None
    assert rg.cut_with_mesh(cut_surface) is not None


def test_cut_with_plane():
    print("\n--- test_cut_with_plane ---")
    _, tm, rg = _make_datasets()
    assert tm.cut_with_plane(normal=(1, 1, 0), origin=(500, 0, 0)) is not None
    assert rg.cut_with_plane(normal=(1, 1, 0), origin=(0, 0, 0)) is not None


def test_extract_cells_by_type():
    print("\n--- test_extract_cells_by_type ---")
    _, tm, _ = _make_datasets()
    out = tm.extract_cells_by_type("tetra")
    assert out is not None
    assert out.ncells > 0


def test_isosurface():
    print("\n--- test_isosurface ---")
    vol, tm, rg = _make_datasets()
    assert vol.isosurface() is not None
    assert tm.isosurface() is not None
    rg.map_cells_to_points()
    assert rg.isosurface() is not None


def _run_as_script():
    tests = sorted(
        (name, func)
        for name, func in globals().items()
        if name.startswith("test_") and callable(func)
    )
    for name, func in tests:
        print(f"running {name}")
        func()


if __name__ == "__main__":
    _run_as_script()
