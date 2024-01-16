"""Probe a Volume (voxel dataset) with lines"""
from vedo import *

vol = Volume(dataurl + "embryo.slc")

lines = []
for i in range(60):  # probe scalars on 60 parallel lines
    step = (i - 30) * 2
    p1 = vol.center() + [-100, step, step]
    p2 = vol.center() + [ 100, step, step]
    ln = Line(p1, p2, res=100)
    lines.append(ln)
lines = merge(lines)

# Probe the Volume with the lines and add the scalars as pointdata
lines.probe(vol)
lines.lw(3).cmap('hot', vmin=0, vmax=110).add_scalarbar()
print(lines)

show(vol, lines, __doc__, axes=1).close()
