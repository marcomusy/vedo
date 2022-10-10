"""Rendering with Screen Space Ambient Occlusion (SSAO)"""
from vedo import dataurl, Mesh, Volume, Plotter

# mesh = Mesh(dataurl + "porsche.ply").rotate_x(90)
mesh = Volume(dataurl+"embryo.tif").isosurface()
mesh.compute_normals().c('white')

plt = Plotter(N=2, bg='blue1')

plt.at(0)
radius = mesh.diagonal_size()/5  # need to specify it!
plt.add_ambient_occlusion(radius)
plt += mesh.clone()
plt += __doc__

plt.at(1)
plt += mesh.clone()
plt += '..without ambient occlusion'

plt.show(viewup='z', zoom=1.3)
plt.interactive().close()
