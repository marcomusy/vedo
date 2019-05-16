"""
Intersect a Volume (voxel dataset) with planes
"""
from vtkplotter import show, load, probePlane, vector, Text, datadir

vol = load(datadir+"embryo.slc")

planes = []
for i in range(6):
    print("probing slice plane #", i)
    pos = vol.imagedata().GetCenter() + vector(0, 0, (i - 3) * 25.0)
    a = probePlane(vol, origin=pos, normal=(0, 0, 1)).alpha(0.2)
    planes.append(a)
    # print(max(a.scalars(0))) # access scalars this way, 0 means first

show(planes, Text(__doc__), axes=4, bg="w")
