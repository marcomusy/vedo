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
g.addChild(0, edgeLabel="Mother giving birth\nto her baby cell")
g.addChild(1); g.addChild(1)
g.addChild(2); g.addChild(2); g.addChild(2)
g.addChild(3); g.addChild(3, edgeLabel="daughter_38")
g.addChild(4); g.addChild(4)
for i in range(7): g.addChild(5, nodeLabel="cell5_"+str(i))
g.addChild(7); g.addChild(7); g.addChild(7)

g.build() # optimize layout

g.unpack(0).color('dg').lineWidth(3) #0=graph, 1=vertexLabels, 2=edgeLabels, 3=arrows
g.unpack(2).color('dr')

show(g, __doc__, axes=9, elevation=-40).close()
