"""
Create a set of transparencies which can be passed to method pointColors()
"""
from vtkplotter import load, show, Text, datadir

mesh = load(datadir+"beethoven.ply")

# pick y coordinates of vertices and use them as scalars
scals = mesh.getPoints()[:, 1]

# define opacities in the range of the scalar,
# at min(scals) alpha is 0.1,
# at max(scals) alpha is 0.9:
alphas = [0.1, 0.1, 0.3, 0.4, 0.9]

mesh.pointColors(scals, alpha=alphas, cmap="copper")
# print(mesh.scalars('pointColors_copper')) # retrieve scalars

show(mesh, Text(__doc__), axes=9)
