"""Rendering with Screen Space Ambient Occlusion (SSAO)"""
from vedo import dataurl, Mesh, Volume, Plotter

# mesh = Mesh(dataurl + "porsche.ply").rotateX(90)
mesh = Volume(dataurl+"embryo.tif").isosurface()
mesh.computeNormals().c('white')

plt = Plotter(N=2, bg='blue1')

plt.at(0)
radius = mesh.diagonalSize()/5  # need to specify it!
plt.addAmbientOcclusion(radius)
plt += mesh.clone()
plt += __doc__

plt.at(1)
plt += mesh.clone()
plt += '..without ambient occlusion'

plt.show(viewup='z', zoom=1.3)
plt.interactive().close()
