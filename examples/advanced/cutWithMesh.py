'''
Cut a mesh with another mesh.
'''
from vtkplotter import *

embryo = load('data/embryo.tif').normalize()

# mesh used to cut:
msh = ellipsoid().pos(.8,.1,-.3).scale(0.5).wire()

cutembryo = embryo.clone().cutWithMesh(msh)

show([embryo, msh, text(__doc__)], at=0, N=2, viewup='z')
show(cutembryo, at=1, interactive=1)


