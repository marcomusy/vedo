from __future__ import annotations

import numpy as np

from vedo.pyplot import violin


def test_violin_unsplined_fill_is_centered_on_x_axis() -> None:
    hist = violin(
        np.array([0.0, 0.0, 1.0, 1.0]),
        bins=2,
        vlim=(0.0, 2.0),
        x=5.0,
        width=2.0,
        splined=False,
        centerline=False,
        c="t",
        lc="k",
    )

    rectangles = [part for part in hist.unpack() if part.name == "Rectangle"]

    assert len(rectangles) == 2
    for rectangle in rectangles:
        xmin, xmax = rectangle.bounds()[:2]
        assert np.isclose((xmin + xmax) / 2, 5.0)
