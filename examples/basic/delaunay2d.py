'''
Example for delaunay2D() and cellCenters() functions.
'''
print(__doc__)

from vtkplotter import Plotter, delaunay2D, Points

vp = Plotter(shape=(1,2), interactive=0)

d0 = vp.load('data/250.vtk', legend='original mesh').rotateY(-90)

coords = d0.coordinates() # get the coordinates of the mesh vertices
# Build a mesh starting from points in space 
#  (points must be projectable on the XY plane)
d1 = delaunay2D(coords)
d1.color('r').wire(True).legend('delaunay mesh')

cents = d1.cellCenters()
ap = Points(cents).legend('cell centers')

vp.show([d0,d1], at=0) # NB: d0 and d1 are slightly different
vp.show([d1,ap], at=1, interactive=1)
