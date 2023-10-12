"""Create a set of transparencies
which can be passed to method cmap()"""
from vedo import Mesh, show, dataurl

mesh = Mesh(dataurl+"beethoven.ply")

# pick y coordinates of vertices and use them as scalars
scalars = mesh.vertices[:, 1]

# define opacities in the range of the scalar,
# at min(scals) alpha is 0.1,
# at max(scals) alpha is 0.9:
alphas = [0.1, 0.1, 0.3, 0.4, 0.9]

mesh.cmap("copper", scalars, alpha=alphas)
# mesh.print()
# print(mesh.pointdata['PointScalars']) # retrieve scalars

show(mesh, __doc__, axes=9).close()
