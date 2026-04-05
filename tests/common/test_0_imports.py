from __future__ import annotations

import numpy as np

import vedo.vtkclasses


def test_vtkclasses_imports() -> None:
    assert vedo.vtkclasses is not None
    assert np.__version__
