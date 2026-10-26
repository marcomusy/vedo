from __future__ import annotations

import numpy as np
import pytest

from vedo.utils import (
    line_line_distance,
    make3d,
    segment_segment_distance,
    triangle_solver,
)


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


def test_triangle_solver_three_sides() -> None:
    # 3-4-5 right triangle: bc is opposite a, ac is opposite b
    t = triangle_solver(a=3, b=4, c=5)[0]
    assert np.isclose(t["ab"], np.pi / 2)
    assert np.isclose(t["bc"], np.arctan2(3, 4))
    assert np.isclose(t["ac"], np.arctan2(4, 3))

    # obtuse angle ac (opposite the longest side b)
    t = triangle_solver(a=1, b=1.9, c=1)[0]
    assert t["ac"] > np.pi / 2
    assert np.isclose(t["ab"] + t["bc"] + t["ac"], np.pi)

    # two sides and the included angle go through the same formulas
    t = triangle_solver(a=3, b=4, ab=np.pi / 2)[0]
    assert np.isclose(t["c"], 5)
    assert np.isclose(t["bc"], np.arctan2(3, 4))
    assert np.isclose(t["ac"], np.arctan2(4, 3))
