"""Cut a mesh with an other mesh
and cap the holes"""
from vedo import *

p1 = Mesh(dataurl+'motor.byu')
cutmesh = Sphere().y(-0.4).scale(0.4).wireframe().alpha(0.1)
show(p1, cutmesh, __doc__, at=0, N=2, axes=1, viewup="z")

p2 = p1.clone().cutWithMesh(cutmesh)
redcap = p2.cap(returnCap=True).color("r", 0.5)
show(redcap, p2, at=1, interactive=True).close()
