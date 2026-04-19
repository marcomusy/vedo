from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vedo import Ellipsoid, Line, RectilinearGrid, Sphere, TetMesh, Volume


def _make_datasets() -> dict[str, object]:
    vol = Volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    rg = RectilinearGrid(
        [
            np.linspace(-1.0, 1.0, 3),
            np.linspace(-1.0, 1.0, 3),
            np.linspace(-1.0, 1.0, 3),
        ]
    )
    rg.pointdata["density"] = np.linspace(0.0, 1.0, rg.npoints)
    rg.celldata["cell_density"] = np.linspace(0.0, 1.0, rg.ncells)

    tm = TetMesh(rg.dataset)
    tm.pointdata["density"] = np.linspace(0.0, 1.0, tm.npoints)
    tm.celldata["cell_density"] = np.linspace(0.0, 1.0, tm.ncells)

    return {"vol": vol, "tm": tm, "rg": rg}


def _make_line() -> Line:
    return Line([0, 0, 0], [1, 1, 1], res=100)


@pytest.fixture
def dataset_bundle() -> dict[str, object]:
    return _make_datasets()


@pytest.mark.parametrize("dataset_name", ["vol", "tm", "rg"], ids=["volume", "tetmesh", "rect-grid"])
def test_add_ids(dataset_bundle, dataset_name: str) -> None:
    dataset = dataset_bundle[dataset_name]
    assert dataset.add_ids() is dataset


@pytest.mark.parametrize("dataset_name", ["vol", "tm", "rg"], ids=["volume", "tetmesh", "rect-grid"])
def test_basic_dataset_properties(dataset_bundle, dataset_name: str) -> None:
    dataset = dataset_bundle[dataset_name]
    assert dataset.average_size() > 0
    assert len(dataset.bounds()) == 6
    assert dataset.cell_centers().coordinates.shape[1] == 3
    assert len(dataset.center_of_mass()) == 3
    assert dataset.compute_cell_size() is dataset
    assert dataset.copy_data_from(dataset.clone()) is not None
    assert isinstance(dataset.lines, list)
    assert isinstance(dataset.lines_as_flat_array, np.ndarray)
    assert dataset.mark_boundaries() is dataset
    assert isinstance(dataset.memory_address(), int) and dataset.memory_address() > 0
    assert dataset.memory_size() > 0
    assert dataset.modified() is dataset
    assert dataset.npoints > 0
    assert dataset.ncells > 0
    assert dataset.tomesh() is not None


@pytest.mark.parametrize(
    ("dataset_name", "expected_cells_len"),
    [("vol", 0), ("tm", None), ("rg", 0)],
    ids=["volume", "tetmesh", "rect-grid"],
)
def test_cells(dataset_bundle, dataset_name: str, expected_cells_len: int | None) -> None:
    dataset = dataset_bundle[dataset_name]
    if expected_cells_len is None:
        assert len(dataset.cells) == dataset.ncells
    else:
        assert len(dataset.cells) == expected_cells_len


def test_find_cells(dataset_bundle) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]
    bounds = Sphere().bounds()

    assert isinstance(vol.find_cells_along_line([0, 0, 0], [10, 10, 10]), np.ndarray)
    assert isinstance(tm.find_cells_along_line([0, 0, 0], [10, 1, 1]), np.ndarray)
    assert isinstance(rg.find_cells_along_line([0, 0, 0], [10, 1, 1]), np.ndarray)

    assert isinstance(vol.find_cells_in_bounds(bounds), np.ndarray)
    assert isinstance(tm.find_cells_in_bounds(bounds), np.ndarray)
    assert isinstance(rg.find_cells_in_bounds(bounds), np.ndarray)


def test_integrate_and_interpolate(dataset_bundle) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]

    assert vol.integrate_data() is not None
    assert tm.integrate_data() is not None
    assert rg.integrate_data() is not None

    assert vol.interpolate_data_from(vol, n=1) is vol
    assert tm.interpolate_data_from(vol, n=1) is tm
    assert rg.interpolate_data_from(vol, n=1) is rg


@pytest.mark.parametrize("dataset_name", ["vol", "tm", "rg"], ids=["volume", "tetmesh", "rect-grid"])
def test_map_between_points_and_cells(dataset_bundle, dataset_name: str) -> None:
    dataset = dataset_bundle[dataset_name]
    assert dataset.clone().map_cells_to_points() is not None
    assert dataset.clone().map_points_to_cells() is not None


def test_probe(dataset_bundle) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]
    line = _make_line()
    assert line.probe(vol) is not None
    assert line.probe(tm) is not None
    assert line.probe(rg) is not None


def test_resample_and_smooth(dataset_bundle) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]
    assert vol.clone().resample_data_from(vol) is not None
    assert tm.clone().resample_data_from(tm) is not None
    assert rg.clone().resample_data_from(rg) is not None
    assert vol.smooth_data() is vol
    assert tm.smooth_data() is tm
    assert rg.smooth_data() is rg


def test_tetmesh_special_cases(dataset_bundle) -> None:
    tm = dataset_bundle["tm"]
    shrunk = tm.shrink()
    assert shrunk is not None
    assert shrunk.ncells == tm.ncells

    extracted = tm.extract_cells_by_type("tetra")
    assert extracted is not None
    assert extracted.ncells > 0


def test_write(dataset_bundle, tmp_path: Path) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]

    vti = tmp_path / "test.vti"
    vtu = tmp_path / "test.vtu"
    vtr = tmp_path / "test.vtr"

    vol.write(str(vti))
    tm.write(str(vtu))
    rg.write(str(vtr))

    assert vti.exists()
    assert vtu.exists()
    assert vtr.exists()


def test_cutting_operations(dataset_bundle) -> None:
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]
    cut_surface = Ellipsoid().scale(5)
    assert tm.clone().cut_with_mesh(cut_surface) is not None
    assert rg.clone().cut_with_mesh(cut_surface) is not None
    assert tm.clone().cut_with_plane(normal=(1, 1, 0), origin=(0, 0, 0)) is not None
    assert rg.clone().cut_with_plane(normal=(1, 1, 0), origin=(0, 0, 0)) is not None


def test_isosurface(dataset_bundle) -> None:
    vol = dataset_bundle["vol"]
    tm = dataset_bundle["tm"]
    rg = dataset_bundle["rg"]
    assert vol.isosurface() is not None
    assert tm.isosurface() is not None
    assert rg.isosurface() is not None
