"""Cut an UnstructuredGrid with a plane"""
from vedo import UnstructuredGrid, dataurl, show

ug = UnstructuredGrid(dataurl+'ugrid.vtk').cmap("jet")
ug = ug.cut_with_plane(origin=(5,0,1), normal=(1,1,5))
show(repr(ug), ug, axes=1, viewup='z').close()

# Create a polygonal Mesh from the UGrid and shrink it
msh = ug.shrink(0.9).tomesh().c('gold',0.2)
show(repr(msh), msh, axes=1, viewup='z').close()
