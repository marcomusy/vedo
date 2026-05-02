from __future__ import annotations

import numpy as np

from vedo.pyplot import Histogram2D


def test_histogram2d_accepts_stacked_xy_format() -> None:
    data = [[1, 2, 3], [4, 5, 6]]

    hist = Histogram2D(
        data,
        bins=(2, 2),
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )

    assert hist.entries == 3
    assert np.allclose(hist.mean, (2.0, 5.0))
    assert np.allclose(hist.xlim, (1.0, 3.0))
    assert np.allclose(hist.ylim, (4.0, 6.0))
    assert hist.frequencies.shape == (2, 2)


def test_histogram2d_accepts_point_pair_format() -> None:
    data = [(1, 4), (2, 5), (3, 6)]

    hist = Histogram2D(
        data,
        bins=(2, 2),
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )

    assert hist.entries == 3
    assert np.allclose(hist.mean, (2.0, 5.0))
    assert np.allclose(hist.xlim, (1.0, 3.0))
    assert np.allclose(hist.ylim, (4.0, 6.0))
    assert hist.frequencies.shape == (2, 2)


def test_histogram2d_data_formats_are_equivalent() -> None:
    stacked = Histogram2D(
        [[1, 2, 3], [4, 5, 6]],
        bins=(2, 2),
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )
    paired = Histogram2D(
        [(1, 4), (2, 5), (3, 6)],
        bins=(2, 2),
        xlim=(None, None),
        ylim=(None, None),
        scalarbar=False,
        axes=False,
    )

    assert np.array_equal(stacked.frequencies, paired.frequencies)
    assert np.allclose(stacked.edges[0], paired.edges[0])
    assert np.allclose(stacked.edges[1], paired.edges[1])
