from __future__ import annotations

import numpy as np
import pytest

from vedo.utils import make3d


@pytest.mark.parametrize(
    ("input_values", "expected"),
    [
        ([], np.array([])),
        ([0, 1], np.array([0, 1, 0])),
        ([[0, 1], [9, 8]], np.array([[0, 1, 0], [9, 8, 0]])),
        (
            [[0, 1], [6, 7], [6, 7], [6, 7]],
            np.array([[0, 1, 0], [6, 7, 0], [6, 7, 0], [6, 7, 0]]),
        ),
        ([0, 1, 2], np.array([0, 1, 2])),
        ([[0, 1, 2]], np.array([[0, 1, 2]])),
        ([[0, 1, 2], [6, 7, 8]], np.array([[0, 1, 2], [6, 7, 8]])),
        (
            [[0, 1, 2], [6, 7, 8], [6, 7, 9]],
            np.array([[0, 1, 2], [6, 7, 8], [6, 7, 9]]),
        ),
        (
            [[0, 1, 2], [6, 7, 8], [6, 7, 8], [6, 7, 4]],
            np.array([[0, 1, 2], [6, 7, 8], [6, 7, 8], [6, 7, 4]]),
        ),
    ],
)
def test_make3d(input_values, expected) -> None:
    result = make3d(input_values)
    assert np.array_equal(result, expected)
