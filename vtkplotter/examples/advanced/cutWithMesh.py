"""Cut a mesh with another mesh"""
from vtkplotter import *

embryo = load(datadir+"embryo.tif", threshold=30).normalize()
txt = Text2D(__doc__, c='b', bg='lb')

# mesh used to cut:
msh = Ellipsoid().pos(0.8, 0.1, -0.3).scale(0.5).wireframe()

# make a working copy and cut it with the ellipsoid
cutembryo = embryo.clone().cutWithMesh(msh).backColor("t")

show(embryo, msh,    at=0, N=2, axes=1, viewup="z")
show(cutembryo, txt, at=1, interactive=1)
