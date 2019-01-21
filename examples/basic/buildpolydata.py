# Load a mesh, extract the vertex coordinates, and build a new vtkPolyData
# object. Faces (vertex connectivity) can be specified too.
# 
from vtkplotter import show, load, buildPolyData


pts = load('data/shapes/bunny.obj').subdivide(N=2).coordinates()

poly = buildPolyData(pts, faces=None) # vtkPolyData made of just vertices

show(poly)  # (press p to increase point size)

