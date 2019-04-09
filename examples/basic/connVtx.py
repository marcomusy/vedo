"""
Find the vertices that are connected 
to a specific vertex in a mesh.
"""
from vtkplotter import *

s = Sphere(c="y", res=12).wire()

index = 12

vtxs = s.connectedVertices(index, returnIds=False)

apt = Point(s.getPoint(index), c="r", r=15)
cpts = Points(vtxs, c="blue", r=15)

show(s, apt, cpts, Text(__doc__), verbose=False)
