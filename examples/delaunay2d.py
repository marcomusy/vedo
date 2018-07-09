# Example of using delaunay2D() and cellCenters()
# 
import plotter

vp = plotter.vtkPlotter(shape=(1,2), interactive=0)

d0 = vp.load('data/250.vtk', edges=1, legend='original mesh').rotateY(-90)

coords = d0.coordinates() # get the coordinates of the mesh vertices
# Build a mesh starting from points in space (they must be projectable on the XY plane)
d1 = vp.delaunay2D(coords, c='r', wire=1, legend='delaunay mesh')

cents = d1.cellCenters()
ap = vp.points(cents, legend='cell centers')

vp.show([d0,d1], at=0) # NB: d0 and d1 are slightly different
vp.show([d1,ap], at=1, interactive=1)
