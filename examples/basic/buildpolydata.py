# Load a mesh, extract the vertex coordinates, and build a new vtkPolyData
# object. Faces (vertex connectivity) can be specified too.
# 
from vtkplotter import Plotter
from vtkplotter.vtkio import buildPolyData


vp = Plotter()

verts = vp.load('data/shapes/bunny.obj').subdivide(N=2).coordinates()

poly = buildPolyData(verts, faces=None) # vtkPolyData

vp.show(poly)  # (press p to increase point size)

