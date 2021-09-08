"""Probe a Volume (voxel dataset) with lines"""
from vedo import *

vol = Volume(dataurl+"embryo.slc")

lines = []
for i in range(60):  # probe scalars on 60 parallel lines
    step = (i - 30) * 2
    p1 = vol.center() + vector(-100, step, step)
    p2 = vol.center() + vector( 100, step, step)
    pl = probeLine(vol, p1, p2).cmap('hot', vmin=0, vmax=110)
    pl.alpha(0.5).lineWidth(4)
    lines.append(pl)
    #print(pl.pointdata.keys()) # numpy scalars can be accessed here
    #print(pl.pointdata['vtkValidPointMask']) # the mask of valid points

show(lines, __doc__, axes=1).close()
