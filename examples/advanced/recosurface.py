"""
Reconstruct a polygonal surface
from a point cloud:

 1. An object is loaded and
    noise is added to its vertices.
 2. The point cloud is smoothened
    with MLS (Moving Least Squares)
 3. Impose a minimum distance among points
 4. A triangular mesh is extracted from
    this set of sparse Points.
"""
from vedo import dataurl, printc, Plotter, Points, Mesh, Text2D


plt = Plotter(shape=(1,5))
plt.at(0).show(Text2D(__doc__, s=0.75, font='Theemim', bg='green5'))

mesh = Mesh(dataurl+"apple.ply").subdivide()
plt.at(1).show(mesh)

pts0 = Points(mesh, r=3).addGaussNoise(1)
plt.at(2).show(pts0)

pts1 = pts0.clone().smoothMLS2D(f=0.8)  # smooth cloud
printc("Nr of points before cleaning nr. points:", pts1.N())

# impose a min distance among mesh points
pts1.subsample(0.005)
printc("             after  cleaning nr. points:", pts1.N())
plt.at(3).show(pts1)

# reconstructed surface from point cloud
reco = pts1.reconstructSurface(dims=100, radius=0.2)
plt.at(4).show(reco, axes=7, zoom=1.2)

plt.interactive().close()
