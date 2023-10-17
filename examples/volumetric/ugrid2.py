"""Cut UGrid with plane"""
from vedo import *

ug = UGrid(dataurl+'ugrid.vtk')
ug.cmap('k5')

ug = ug.cut_with_plane(origin=(5,0,1), normal=(1,1,5))

msh = ug.tomesh(shrink=0.8) # return a polygonal Mesh

show([(ug, __doc__), msh], N=2, axes=1, viewup='z').close()
