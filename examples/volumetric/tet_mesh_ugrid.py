"""Read a tetrahedral mesh from a 
vtkUnstructuredGrid object and visualize 
it as either a Volume or a mesh Actor
"""
from vtkplotter import *

ug = loadUnStructuredGrid(datadir+'limb_ugrid.vtk')

cmap = 'nipy_spectral'

vol = Volume(ug).color(cmap)

################################
# if False will only show the outer surface:
settings.visibleGridEdges = True

mesh = Actor(ug).color(cmap).alpha(0.2).addScalarBar(c='w')

txt = Text(__doc__, c='w')

show([(txt,vol), mesh], N=2)
