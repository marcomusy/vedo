"""
Example to show how to use recoSurface()
to reconstruct a surface from points.

 1. An object is loaded and
    noise is added to its vertices.
 2. the point cloud is smoothened with MLS
    (see moving_least_squares.py)
 3. mesh.clean() imposes a minimum distance
    among mesh points where 'tol' is the
    fraction of the mesh size.
 4. a triangular mesh is extracted from
    this set of sparse Points, 'bins' is the
    number of voxels of the subdivision
"""
print(__doc__)
from vedo import *
import numpy as np


plt = Plotter(N=4, axes=0)

mesh = plt.load(dataurl+"apple.ply").subdivide()
plt.show(mesh, at=0)

noise = np.random.randn(mesh.N(), 3) * 0.03

pts0 = Points(mesh.points() + noise, r=3).legend("noisy cloud")
plt.show(pts0, at=1)

pts1 = pts0.clone().smoothMLS2D(f=0.8)  # smooth cloud

print("Nr of points before cleaning nr. points:", pts1.N())

# impose a min distance among mesh points
pts1.clean(tol=0.005).legend("smooth cloud")
print("             after  cleaning nr. points:", pts1.N())

plt.show(pts1, at=2)

# reconstructed surface from point cloud
reco = recoSurface(pts1, dims=100, radius=0.2).legend("surf. reco")
plt.show(reco, at=3, axes=7, zoom=1.2, interactive=1).close()
