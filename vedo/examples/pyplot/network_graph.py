"""Visualize a 2D/3D network and its properties"""
# (see also example: lineage_graph.py)
from vedo import Points, show, sin
from vedo.pyplot import DirectedGraph

# Create some graph with nodes and edges
# layouts: [2d, fast2d, clustering2d, circular, circular3d, cone, force, tree]
g = DirectedGraph(layout='fast2d')
for i in range(6): g.addChild(i)  # add one child node to node i
for i in range(3): g.addChild(i)
for i in range(3): g.addChild(i)
for i in range(7,9): g.addChild(i)
for i in range(3): g.addChild(12) # add 3 children to node 12
g.build()

dgraph = g.unpack(0).lineWidth(4) # get the graph lines
nodes = dgraph.points()           # get the 3d points of the nodes

pts = Points(nodes, r=12).c('red').alpha(0.5)

v1 = ['node'+str(n) for n in range(len(nodes))]
v2 = [sin(x) for x in range(len(nodes))]
labs1 = pts.labels(v1, scale=.04, italic=True).addPos(.05,0.04,0).c('green')
labs2 = pts.labels(v2, scale=.04, precision=3).addPos(.05,-.04,0).c('red')

# Interpolate the node value to color the edges:
dgraph.clean().pointColors(v2, cmap='viridis').addScalarBar()

# This would colorize the edges directly with solid color based on the v3 array:
# v3 = [sin(x) for x in range(dgraph.NCells())]
# dgraph.cellColors(v3, cmap='jet').addScalarBar()

show(pts, dgraph, labs1, labs2, __doc__, axes=9)
