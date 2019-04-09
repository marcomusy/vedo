"""
Cut a paraboloid with a mesh and cap the holes.
"""
from vtkplotter import *

p1 = Paraboloid().rotateX(90)

cutmesh = Hyperboloid().scale(0.4).wire(True).alpha(0.1)

show(p1, cutmesh, at=0, N=2, axes=1, viewup="z")

p2 = p1.clone().cutWithMesh(cutmesh)

redcap = p2.cap(returnCap=True).color("r")  # dark red cap only

show(redcap, p2, Text(__doc__), at=1, interactive=1)
