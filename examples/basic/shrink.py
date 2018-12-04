# Shrink the triangulation of a mesh to make the inside visible
#
from vtkplotter import Plotter


vp = Plotter()

vp.load('data/shapes/teapot.vtk').shrink(0.8)

vp.sphere(r=0.2).pos([0,0,-0.5])

vp.show(viewup='z', zoom=1.2)

