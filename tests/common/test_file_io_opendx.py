#!/usr/bin/env python3
from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import vedo
from vedo.file_io.loaders import loadImageData
from vedo.utils import vtk2numpy


def test_load_dx_returns_image_data_and_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    origin = np.array([1.0, 2.0, 3.0], dtype=float)
    delta = np.array([0.5, 1.5, 2.5], dtype=float)

    class FakeGrid:
        def __init__(self, grid, file_format=None) -> None:
            assert str(grid).endswith(".dx")
            assert file_format == "dx"
            self.grid = data
            self.origin = origin
            self.delta = delta
            self.metadata = {"source": "fake"}

    fake_module = types.ModuleType("gridData")
    fake_module.Grid = FakeGrid
    monkeypatch.setitem(sys.modules, "gridData", fake_module)

    dx_file = tmp_path / "sample.dx"
    dx_file.write_text("object 1 class gridpositions counts 2 3 4\n", encoding="utf-8")

    image = loadImageData(dx_file)
    assert image is not None
    assert image.GetClassName() == "vtkImageData"
    assert image.GetDimensions() == data.shape
    assert image.GetOrigin() == pytest.approx(tuple(origin))
    assert image.GetSpacing() == pytest.approx(tuple(delta))
    assert np.array_equal(vtk2numpy(image.GetPointData().GetScalars()), data.ravel(order="F"))

    volume = vedo.load(dx_file)
    assert isinstance(volume, vedo.Volume)
    assert tuple(volume.dimensions()) == data.shape
    assert volume.dataset.GetOrigin() == pytest.approx(tuple(origin))
    assert volume.dataset.GetSpacing() == pytest.approx(tuple(delta))


def test_load_dx_prompts_for_griddataformats_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dx_file = tmp_path / "missing-dependency.dx"
    dx_file.write_text("object 1 class gridpositions counts 2 2 2\n", encoding="utf-8")

    monkeypatch.delitem(sys.modules, "gridData", raising=False)

    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "gridData":
            raise ModuleNotFoundError("No module named 'gridData'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    with pytest.raises(ImportError, match="pip install gridDataFormats"):
        loadImageData(dx_file)
