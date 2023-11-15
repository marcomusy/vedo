"""Dijkstra algorithm to compute the graph geodesic.
Take as input a polygonal mesh and perform
a shortest path calculation between two vertices."""
from vedo import IcoSphere, Earth, show


msh = IcoSphere(r=1.02, subdivisions=4)
msh.wireframe().alpha(0.2)

path = msh.geodesic([0.349,-0.440,0.852], [-0.176,-0.962,0.302])
# path = msh.geodesic(36, 442) # use vertex indices

# printc(geo.pointdata["VertexIDs"])

show(Earth(), msh, path, __doc__, bg2='lb', viewup="z", zoom=1.3).close()
