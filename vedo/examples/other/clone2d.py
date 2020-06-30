"""Make a 2D copy of a Mesh and place it in the 3D scene"""
from vedo import load, datadir, show

s = load(datadir+'man.vtk').rotateZ(20).rotateX(-70).scale(0.2)
s.c('darkgreen').alpha(0.3)

# Make a sort of 2D snapshot of the mesh (return vtkActor2D)
# The coordinate system options are
#     0. Displays
#     1. Normalized Display
#     2. Viewport (origin is the bottom-left corner of the window)
#     3. Normalized Viewport
#     4. View (origin is the center of the window)
#     5. World (anchor the 2d image to mesh)
s2d = s.clone2D(pos=[0.4,0.4], coordsys=4, c='r', alpha=1)

show(s, s2d, __doc__, axes=1)

