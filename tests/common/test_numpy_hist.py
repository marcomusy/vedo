from __future__ import annotations

import numpy as np


def test_numpy_histogram2d_accepts_auto_x_range_with_explicit_y_range() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = np.array([10.0, 11.0, 12.0, 13.0])
    yrange = [10.0, 13.0]

    hist, xedges, yedges = np.histogram2d(x, y, bins=(3, 3), range=(None, yrange))
    expected, expected_xedges, expected_yedges = np.histogram2d(
        x, y, bins=(3, 3), range=([x.min(), x.max()], yrange)
    )

    assert np.array_equal(hist, expected)
    assert np.allclose(xedges, expected_xedges)
    assert np.allclose(yedges, expected_yedges)
