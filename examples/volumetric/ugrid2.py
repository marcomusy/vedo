"""Cut an UnstructuredGrid with a plane"""
from vedo import *

ug = UGrid(dataurl+'ugrid.vtk')
ug.cmap('blue8')

ug = ug.cut_with_plane(origin=(5,0,1), normal=(1,1,5))

# Create a polygonal Mesh for visualization
msh = ug.shrink(0.9).tomesh()

# note the difference with the following:
# msh = ug.tomesh().shrink(0.9)

show(msh, axes=1, viewup='z').close()
