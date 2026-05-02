from __future__ import annotations

import numpy as np

from vedo.pyplot import Histogram2D


def test_histogram2d_default_bins_from_xy_arrays() -> None:
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(10.0, 20.0, 11)

    hist = Histogram2D(
        x,
        y,
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )

    assert hist.entries == len(x)
    assert hist.frequencies.shape == (25, 25)
    assert hist.frequencies.sum() == len(x)
    assert np.allclose(hist.mean, (0.0, 15.0))
    assert np.allclose(hist.xlim, (-2.0, 2.0))
    assert np.allclose(hist.ylim, (10.0, 20.0))
