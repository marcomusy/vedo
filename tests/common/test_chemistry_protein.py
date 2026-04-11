#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import vedo
from vtkmodules.util.numpy_support import vtk_to_numpy
from vedo.applications.chemistry import Protein


MINI_PDB = """\
ATOM      1  N   GLY A   1      11.104  13.207  10.000  1.00 20.00           N
ATOM      2  CA  GLY A   1      12.560  13.200  10.000  1.00 20.00           C
ATOM      3  C   GLY A   1      13.000  14.620  10.000  1.00 20.00           C
ATOM      4  O   GLY A   1      12.300  15.580  10.000  1.00 20.00           O
ATOM      5  N   ALA A   2      14.220  14.760  10.000  1.00 20.00           N
ATOM      6  CA  ALA A   2      14.780  16.100  10.000  1.00 20.00           C
ATOM      7  C   ALA A   2      16.280  16.020  10.000  1.00 20.00           C
ATOM      8  O   ALA A   2      16.960  17.020  10.000  1.00 20.00           O
TER
END
"""


def _write_pdb(tmp_path: Path) -> Path:
    pdb_path = tmp_path / "mini_protein.pdb"
    pdb_path.write_text(MINI_PDB, encoding="utf-8")
    return pdb_path


def test_protein_loads_local_pdb(tmp_path: Path) -> None:
    protein = Protein(str(_write_pdb(tmp_path)))
    dataset = protein.actor.GetMapper().GetInput()

    assert dataset is not None
    assert dataset.GetClassName() == "vtkPolyData"
    assert dataset.GetNumberOfPoints() > 0
    assert dataset.GetNumberOfCells() > 0

    rgb_array = protein.filter.GetOutput().GetPointData().GetArray("RGB")
    assert rgb_array is not None
    assert vtk_to_numpy(rgb_array).max() < 255


def test_protein_downloads_https_pdb(monkeypatch, tmp_path: Path) -> None:
    pdb_path = _write_pdb(tmp_path)

    def _fake_download(url: str, force: bool = False, verbose: bool = True) -> str:
        assert url == "https://example.org/mini_protein.pdb"
        return str(pdb_path)

    monkeypatch.setattr(vedo.file_io, "download", _fake_download)

    protein = Protein("https://example.org/mini_protein.pdb")
    dataset = protein.actor.GetMapper().GetInput()

    assert dataset is not None
    assert dataset.GetNumberOfPoints() > 0
    assert dataset.GetNumberOfCells() > 0
