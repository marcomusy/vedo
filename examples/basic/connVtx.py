'''
Find the vertices that are connected to a specific vertex in a mesh.
'''
from vtkplotter import *

s = sphere(c='y', res=12).wire()

index = 12

vtxs = s.connectedVertices(index, returnIds=False)

apt = point(s.point(index), c='r', r=15)
cpts = points(vtxs, c='blue', r=15)

show([s, apt, cpts, text(__doc__)], verbose=False)
