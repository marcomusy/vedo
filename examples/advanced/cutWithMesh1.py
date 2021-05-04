"""Cut a mesh with another mesh"""
from vedo import *

settings.tiffOrientationType = 4 # data origin is bottom-left

embryo = Volume(dataurl+"embryo.tif").isosurface(30).normalize()

# mesh used to cut:
msh = Ellipsoid().scale(0.4).pos(2.8, 1.5, 1.5).wireframe()

# make a working copy and cut it with the ellipsoid
cutembryo = embryo.clone().cutWithMesh(msh).c("gold").bc("t")

show(embryo, msh,    at=0, N=2, axes=1, viewup="z")
show(cutembryo, __doc__, at=1, interactive=True).close()
