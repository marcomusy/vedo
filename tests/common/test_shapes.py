from vedo import Arc, vtk_version
import numpy as np


def test_arc_construction() -> None:
    assert vtk_version
    arc = Arc(
        center=None,
        point1=(1, 1, 1),
        point2=None,
        normal=(0, 0, 1),
        angle=np.pi,
    )
    assert isinstance(arc, Arc)
