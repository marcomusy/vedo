#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from rich.console import Console

import vedo
import vedo.vtkclasses as vtki


def main() -> None:
    q = vedo.Quaternion.from_axis_angle(90, (0, 0, 1))
    print("vtk quaternion type", type(q.T))
    assert isinstance(q.T, vtki.vtkQuaterniond)

    print("rotate", q.rotate([1, 0, 0]))
    assert np.allclose(q.rotate([1, 0, 0]), [0, 1, 0], atol=1e-6)

    print("xyzw", q.xyzw)
    q_xyzw = vedo.Quaternion.from_xyzw(q.xyzw)
    assert np.allclose(q_xyzw.matrix3x3, q.matrix3x3, atol=1e-6)

    q_from_matrix = vedo.Quaternion(q.matrix3x3)
    print("matrix3x3", q_from_matrix.matrix3x3)
    assert np.allclose(q_from_matrix.matrix3x3, q.matrix3x3, atol=1e-6)

    lt = q.to_transform()
    print("linear transform", lt.transform_point([1, 0, 0]))
    assert np.allclose(lt.transform_point([1, 0, 0]), [0, 1, 0], atol=1e-6)

    q_half = vedo.Quaternion().slerp(
        0.5, vedo.Quaternion.from_axis_angle(180, (0, 0, 1))
    )
    print("slerp", q_half.rotate([1, 0, 0]))
    assert np.allclose(q_half.rotate([1, 0, 0]), [0, 1, 0], atol=1e-6)

    print(
        "camera", vedo.camera_from_quaternion([1, 2, 3], [0, 0, 0, 1]).GetFocalPoint()
    )
    assert np.allclose(
        vedo.camera_from_quaternion([1, 2, 3], [0, 0, 0, 1]).GetFocalPoint(), [1, 2, 3]
    )

    console = Console(record=True, force_terminal=False, width=100)
    console.print(q_from_matrix)
    rich_text = console.export_text()
    print(rich_text.strip())
    assert "q (wxyz)" in rich_text
    assert "angle" in rich_text

    lt_console = Console(record=True, force_terminal=False, width=100)
    lt_console.print(vedo.LinearTransform().translate([1, 2, 3]))
    lt_rich_text = lt_console.export_text()
    print(lt_rich_text.strip())
    assert "matrix 4x4" in lt_rich_text
    assert "concatenations" in lt_rich_text

    nlt_console = Console(record=True, force_terminal=False, width=100)
    nlt_console.print(vedo.NonLinearTransform())
    nlt_rich_text = nlt_console.export_text()
    print(nlt_rich_text.strip())
    assert "sources" in nlt_rich_text
    assert "targets" in nlt_rich_text

    tri_console = Console(record=True, force_terminal=False, width=100)
    tri_console.print(vedo.TransformInterpolator("linear"))
    tri_rich_text = tri_console.export_text()
    print(tri_rich_text.strip())
    assert "ntransforms" in tri_rich_text
    assert "trange" in tri_rich_text


if __name__ == "__main__":
    main()
