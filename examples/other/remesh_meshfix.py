#Credits:
#M. Attene. A lightweight approach to repairing digitized polygon meshes. 
#The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
#http://pymeshfix.pyvista.org
#
# pip install pymeshfix
#
from pyvista import wrap
from pymeshfix import MeshFix
from vtkplotter import *

amesh = load(datadir+'270.vtk').flat()

meshfix = MeshFix(wrap(amesh.polydata()))
meshfix.repair()
repaired = meshfix.mesh

#write(repaired, 'repaired.vtk')
show(amesh, repaired, axes=1, N=2)