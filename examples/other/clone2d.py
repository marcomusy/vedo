"""Make a static 2D copy of a mesh
and place it in the rendering window"""
from vedo import Mesh, dataurl, show

man3d = Mesh(dataurl+'man.vtk').rotateZ(20).rotateX(-70).scale(0.2)
man3d.c('darkgreen').lighting('glossy')

# Make a 2D snapshot of a 3D mesh
# The coordinate system options are
#     0. Displays
#     1. Normalized Display
#     2. Viewport (origin is the bottom-left corner of the window)
#     3. Normalized Viewport
#     4. View (origin is the center of the window)
#     5. World (anchor the 2d image to mesh)
# (returns a vtkActor2D)

man2d = man3d.clone2D(pos=[0.4,0.4], coordsys=4, c='r', alpha=1)

show(man3d, man2d, __doc__, axes=1).close()

