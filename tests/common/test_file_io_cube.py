#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pytest
import vedo
from vedo.file_io.loaders import loadGaussianCube, loadImageData

@pytest.mark.skip(reason="test data file not available")
def test_load_cube_returns_image_data_and_volume() -> None:

    image = loadImageData(vedo.dataurl+"methane-den.cube")
    assert image is not None
    assert image.GetClassName() == "vtkImageData"
    assert image.GetDimensions() == (80, 80, 80)
    assert image.GetPointData().GetScalars() is not None

    volume = vedo.load(vedo.dataurl+"methane-den.cube")
    assert isinstance(volume, vedo.Volume)
    assert tuple(volume.dimensions()) == (80, 80, 80)

    poly, cube_volume = loadGaussianCube(vedo.dataurl+"methane-den.cube", b_scale=20, hb_scale=20)
    assert isinstance(poly, vedo.Mesh)
    assert poly.npoints == 5
    assert isinstance(cube_volume, vedo.Volume)
    assert tuple(cube_volume.dimensions()) == (80, 80, 80)
    assert "atom_type" in poly.pointdata.keys()
