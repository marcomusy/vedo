#!/usr/bin/env python3
from __future__ import annotations
"""Minimal non-rendering smoke checks for import graph regressions."""

import numpy as np

import vedo


def main() -> None:
    pts = vedo.Points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    msh = vedo.Sphere(res=12).compute_normals()
    vol = vedo.Volume(np.zeros((8, 8, 8)))
    assert pts.npoints == 2
    assert msh.npoints > 0
    assert vol.npoints > 0
    print("smoke ok")


if __name__ == "__main__":
    main()
