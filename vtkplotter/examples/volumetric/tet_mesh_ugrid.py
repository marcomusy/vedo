"""Read a tetrahedral mesh from a
vtkUnstructuredGrid object and visualize
it as either a Volume or a Mesh
"""
from vtkplotter import *

ug = loadUnStructuredGrid(datadir+'limb_ugrid.vtk')

cmap = 'nipy_spectral'

vol = Volume(ug).color(cmap)

################################
# if False will only show the outer surface:
settings.visibleGridEdges = True

mesh = Mesh(ug).color(cmap).alpha(0.2).addScalarBar(c='w')

txt = Text2D(__doc__, c='w')

show([(txt,vol), mesh], N=2, bg='bb')
