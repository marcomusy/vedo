"""Find the vertices that are connected
to a specific vertex in a mesh
"""
from vedo import *

s = Sphere(c="y", res=12).wireframe()

index = 12 # pick one point
pt = s.points(index)

vtxs = s.connectedVertices(index, returnIds=False)

apt  = Point(pt, c="r", r=15)
cpts = Points(vtxs, c="blue", r=15)

show(s, apt, cpts, __doc__, bg='bb').close()
