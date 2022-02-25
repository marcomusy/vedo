"""Dijkstra algorithm to compute the graph geodesic.

Takes as input a polygonal mesh and performs
a shortest path calculation 20 times"""
from vedo import Sphere, Earth, show

msh = Sphere(r=1.02, res=200).subsample(0.007).wireframe().alpha(0.1)
# msh.triangulate().clean() # often needed!

path = msh.geodesic([0.349,-0.440,0.852], [-0.176,-0.962,0.302]).c("red4")
# path = msh.geodesic(10728, 9056).c("red4") # use vertex indices

# printc(geo.pointdata["VertexIDs"])

show(Earth(), msh, __doc__, path, bg2='lb', viewup="z", zoom=1.3).close()
