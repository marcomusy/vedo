"""Visualize a 2D/3D network and its properties"""
# (see also example: lineage_graph.py)
from vedo import Points, show, sin
from vedo.pyplot import DirectedGraph

# Create some graph with nodes and edges
# layouts: [2d, fast2d, clustering2d, circular, circular3d, cone, force, tree]
g = DirectedGraph(layout='fast2d')

##################### Use networkx to create random nodes and edges
# import networkx
# G = networkx.gnm_random_graph(n=20, m=35)
# for i, j in G.edges(): g.addEdge(j,i)

##################### Manually create nodes and edges
for i in range(6): g.addChild(i)  # add one child node to node i
for i in range(3): g.addChild(i)
for i in range(3): g.addChild(i)
for i in range(7,9): g.addChild(i)
for i in range(3): g.addChild(12) # add 3 children to node 12
g.addEdge(1,16)

##################### build and draw
graph = g.build().unpack(0).lineWidth(4) # get the vedo 3d graph lines
nodes = graph.points()                   # get the 3d points of the nodes

pts = Points(nodes, r=10).lighting('off')

v1 = ['node'+str(n) for n in range(len(nodes))]
v2 = [sin(x) for x in range(len(nodes))]
labs1 = pts.labels(v1, scale=.02, italic=True).shift(.05,0.02,0).c('green')
labs2 = pts.labels(v2, scale=.02, precision=3).shift(.05,-.02,0).c('red')

# Interpolate the node value to color the edges:
graph.cmap('viridis', v2).addScalarBar3D(c='k')
graph.scalarbar.shift(.3,0,0)
pts.cmap('viridis', v2)

# This would colorize the edges directly with solid color based on a v3 array:
# v3 = [sin(x) for x in range(graph.NCells())]
# graph.cmap('jet', v3).addScalarBar()

show(pts, graph, labs1, labs2, __doc__, axes=9, mode="image").close()
