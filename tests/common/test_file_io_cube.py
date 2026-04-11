#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import vedo
from vedo.file_io.loaders import loadGaussianCube, loadImageData


def test_load_cube_returns_image_data_and_volume() -> None:
    cube_path = Path(__file__).resolve().parents[2] / "develop" / "methane-den.cube"

    image = loadImageData(cube_path)
    assert image is not None
    assert image.GetClassName() == "vtkImageData"
    assert image.GetDimensions() == (80, 80, 80)
    assert image.GetPointData().GetScalars() is not None

    volume = vedo.load(cube_path)
    assert isinstance(volume, vedo.Volume)
    assert tuple(volume.dimensions()) == (80, 80, 80)

    poly, cube_volume = loadGaussianCube(cube_path, b_scale=20, hb_scale=20)
    assert isinstance(poly, vedo.Mesh)
    assert poly.npoints == 5
    assert isinstance(cube_volume, vedo.Volume)
    assert tuple(cube_volume.dimensions()) == (80, 80, 80)
    assert "atom_type" in poly.pointdata.keys()
