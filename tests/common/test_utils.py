from __future__ import annotations

import numpy as np
import pytest

from vedo.utils import line_line_distance, make3d, segment_segment_distance


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


def test_line_line_distance() -> None:
    d, pt1, pt2, t1, t2 = line_line_distance([0, 0, 0], [1, 0, 0], [0, 5, 1], [0, 6, 1])
    assert np.isclose(d, 1.0)
    assert np.allclose(pt1, [0, 0, 0])
    assert np.allclose(pt2, [0, 0, 1])
    assert np.isclose(t1, 0.0)
    assert np.isclose(t2, -5.0)


def test_segment_segment_distance() -> None:
    d, pt1, pt2, t1, t2 = segment_segment_distance(
        [0, 0, 0], [1, 0, 0], [0.5, -1, 1], [0.5, 1, 1]
    )
    assert np.isclose(d, 1.0)
    assert np.allclose(pt1, [0.5, 0, 0])
    assert np.allclose(pt2, [0.5, 0, 1])
    assert np.isclose(t1, 0.5)
    assert np.isclose(t2, 0.5)
