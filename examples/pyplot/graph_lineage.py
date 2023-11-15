"""Generate a lineage graph
of cell divisions"""
# N.B.: no positions are specified here, only connectivity!
from vedo import show
from vedo.pyplot import DirectedGraph

# Layouts: [2d, fast2d, clustering2d, circular, circular3d, cone, force, tree]
#g = Graph(layout='2d', zrange=7)
g = DirectedGraph(layout='cone')
#g = DirectedGraph(layout='circular3d', height=1, radius=1.5)
#g = DirectedGraph(layout='force')

# Vertex generation is automatic,
#  add a child to vertex0, so that now vertex1 exists
g.add_child(0, edge_label="Mother cell")
g.add_child(1); g.add_child(1)
g.add_child(2); g.add_child(2); g.add_child(2)
g.add_child(3); g.add_child(3, edge_label="daughter_38")
g.add_child(4); g.add_child(4)
for i in range(7): g.add_child(5, node_label="cell5_"+str(i))
g.add_child(7); g.add_child(7); g.add_child(7)

g.build() # optimize layout

g[0].color('dg').lw(3) #0=graph, 1=vertexLabels, 2=edge_labels, 3=arrows
g[2].color('dr')

show(g, __doc__, axes=9, elevation=-40).close()
