"""
Intersect a Volume (voxel dataset) with planes.
"""
from vtkplotter import show, loadVolume, probeLine, vector, Text, datadir

vol = loadVolume(datadir+"embryo.slc")

pos = vol.imagedata().GetCenter()

lines = []
for i in range(60):  # probe scalars on 60 parallel lines
    step = (i - 30) * 2
    p1, p2 = pos + vector(-100, step, step), pos + vector(100, step, step)
    a = probeLine(vol, p1, p2, res=200)
    a.alpha(0.5).lineWidth(6)
    lines.append(a)
    # print(a.scalars(0)) # numpy scalars can be access here
    # print(a.scalars('vtkValidPointMask')) # the mask of valid points

show(lines, Text(__doc__), axes=4, bg="w")
