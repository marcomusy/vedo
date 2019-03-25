"""
Cut a mesh with another mesh.
Try command line
> vtkplotter data/embryo.tif
to see the threshold range.
"""
from vtkplotter import *

embryo = load(datadir+"embryo.tif", threshold=30).normalize()

# mesh used to cut:
msh = Ellipsoid().pos(0.8, 0.1, -0.3).scale(0.5).wire()

# make a working copy and cut it with the ellipsoid
cutembryo = embryo.clone().cutWithMesh(msh).backColor("t").phong()

show([embryo, msh, Text(__doc__)], at=0, N=2, axes=1, viewup="z")
show(cutembryo, at=1, interactive=1)
