#Credits:
#M. Attene. A lightweight approach to repairing digitized polygon meshes. 
#The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
#http://pymeshfix.pyvista.org
#
# pip install pymeshfix
#
from pymeshfix import MeshFix
from vedo import *

amesh = load(dataurl+'270.vtk')

meshfix = MeshFix(amesh.points(), amesh.faces())
meshfix.repair()
repaired = meshfix.mesh

#write(repaired, 'repaired.vtk')
show(amesh, repaired, axes=1, N=2)