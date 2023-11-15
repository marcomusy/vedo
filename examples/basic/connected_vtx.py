"""Find the vertices that are connected
to a specific vertex in a mesh"""
from vedo import *

# create a wireframe sphere and color it yellow
s = Sphere(res=12).wireframe().c("yellow")

# select one point on the sphere using its index
index = 12
pt = s.vertices[index]

# find all the vertices that are connected to the selected point
ids = s.connected_vertices(index)
vtxs = s.vertices[ids]

# create a red point at the selected point's location
apt = Point(pt).c("red5").ps(15)

# create blue points at the locations of the vertices 
# connected to the selected point
cpts = Points(vtxs).c("blue5").ps(20)

# show the sphere, the selected point, and the connected vertices
show(s, apt, cpts, __doc__, bg='bb').close()
