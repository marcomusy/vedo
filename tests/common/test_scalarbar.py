from __future__ import annotations

import numpy as np

from vedo.pyplot import Histogram2D


def test_histogram2d_scalarbar_adds_scalarbar_actor() -> None:
    x = np.linspace(-1.0, 1.0, 10)
    y = np.linspace(0.0, 2.0, 10)

    hist = Histogram2D(
        x,
        y,
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=True,
        gap=0.1,
        axes=False,
    )
    parts = hist.unpack()

    assert hist.name == "Histogram2D"
    assert hist.entries == len(x)
    assert len(parts) == 2
    assert parts[0].name == "ScalarBar3D"
    assert parts[1].name == "Grid"
