from __future__ import annotations

import numpy as np

from vedo.pyplot import Histogram2D


def test_histogram2d_accepts_explicit_bin_edges() -> None:
    x = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 1.99])
    y = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 1.99])
    xedges = np.array([0.0, 1.0, 2.0])
    yedges = np.array([0.0, 1.0, 2.0])

    hist = Histogram2D(
        x,
        y,
        bins=[xedges, yedges],
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )

    assert hist.entries == len(x)
    assert np.array_equal(hist.edges[0], xedges)
    assert np.array_equal(hist.edges[1], yedges)
    assert np.array_equal(hist.frequencies, [[3.0, 0.0], [0.0, 3.0]])
    assert hist.frequencies.shape == (2, 2)
