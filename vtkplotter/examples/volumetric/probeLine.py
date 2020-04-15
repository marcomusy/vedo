"""Probe a Volume (voxel dataset) with lines."""
from vtkplotter import *

vol = load(datadir+"embryo.slc")

pos = vol.imagedata().GetCenter()

lines = []
for i in range(60):  # probe scalars on 60 parallel lines
    step = (i - 30) * 2
    p1 = pos + vector(-100, step, step)
    p2 = pos + vector( 100, step, step)
    pl = probeLine(vol, p1, p2)
    pl.alpha(0.5).lineWidth(5)
    lines.append(pl)
    # print(pl.getPointArray(0)) # numpy scalars can be access here
    # print(pl.getPointArray('vtkValidPointMask')) # the mask of valid points

show(lines, Text2D(__doc__))
