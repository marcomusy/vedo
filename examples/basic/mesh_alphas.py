"""Create a set of transparencies
which can be passed to method cmap()
"""
from vedo import load, show, datadir

mesh = load(datadir+"beethoven.ply")

# pick y coordinates of vertices and use them as scalars
scals = mesh.points()[:, 1]

# define opacities in the range of the scalar,
# at min(scals) alpha is 0.1,
# at max(scals) alpha is 0.9:
alphas = [0.1, 0.1, 0.3, 0.4, 0.9]

mesh.cmap("copper", scals, alpha=alphas)
# print(mesh.getPointArray('PointScalars')) # retrieve scalars

show(mesh, __doc__, axes=9)
