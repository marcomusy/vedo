"""delaunay2D() and cellCenters() functions"""
from vedo import Plotter, delaunay2D, Points, datadir

vp = Plotter(shape=(1, 2), interactive=0)

d0 = vp.load(datadir+"250.vtk").rotateY(-90).legend("original mesh")

coords = d0.points()  # get the coordinates of the mesh vertices
# Build a mesh starting from points in space
#  (points must be projectable on the XY plane)
d1 = delaunay2D(coords, mode='fit')
d1.color("r").wireframe(True).legend("delaunay mesh")

cents = d1.cellCenters()
ap = Points(cents).legend("cell centers")

# NB: d0 and d1 are slightly different
vp.show(d0, d1, __doc__, at=0, axes=11)
vp.show(d1, ap, at=1, interactive=1)
