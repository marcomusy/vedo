"""
Example to show how to use recoSurface()
to reconstruct a surface from points.

 1. An object is loaded and
    noise is added to its vertices.
 2. The point cloud is smoothened
    with MLS (Moving Least Squares)
 3. Impose a minimum distance among points
 4. A triangular mesh is extracted from
    this set of sparse Points.
"""
from vedo import *


plt = Plotter(shape=(1,5))
plt.show(Text2D(__doc__, s=0.75, font='Theemim', bg='green5'), at=0)

mesh = Mesh(dataurl+"apple.ply").subdivide()
plt.show(mesh, at=1)

pts0 = Points(mesh, r=3).addGaussNoise(1)
plt.show(pts0, at=2)

pts1 = pts0.clone().smoothMLS2D(f=0.8)  # smooth cloud
printc("Nr of points before cleaning nr. points:", pts1.N())

# impose a min distance among mesh points
pts1.subsample(0.005)
printc("             after  cleaning nr. points:", pts1.N())
plt.show(pts1, at=3)

# reconstructed surface from point cloud
reco = recoSurface(pts1, dims=100, radius=0.2)
plt.show(reco, at=4, axes=7, zoom=1.2).interactive().close()
