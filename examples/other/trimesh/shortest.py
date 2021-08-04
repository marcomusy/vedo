import trimesh
import networkx as nx

# test on a sphere mesh
mesh = trimesh.primitives.Sphere()

# edges without duplication
edges = mesh.edges_unique

# the actual length of each unique edge
length = mesh.edges_unique_length

# create the graph with edge attributes for length
g = nx.Graph()
for edge, L in zip(edges, length):
    g.add_edge(*edge, length=L)

# alternative method for weighted graph creation
ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

# arbitrary indices of mesh.vertices to test with
start = 0
end = int(len(mesh.vertices) / 2.0)

# run the shortest path query using length for edge weight
path = nx.shortest_path(g, source=start, target=end, weight="length")

################################### VISUALIZE RESULT
from vedo import show

# make the sphere transparent-ish
mesh.visual.face_colors = [150, 150, 180, 255]

# Path3D with the path between the points
path_visual = trimesh.load_path(mesh.vertices[path])

# visualizable two points
points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])

txt = 'Shortest path query\nusing length for edge weight'

show(mesh, points_visual, path_visual, txt, axes=6).close()
