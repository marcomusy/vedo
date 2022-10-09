"""Cut a mesh with an other mesh and cap the holes"""
from vedo import dataurl, Plotter, Mesh, Sphere

msh1 = Mesh(dataurl+'motor.byu')
cutmesh = Sphere().y(-0.4).scale(0.4).wireframe().alpha(0.1)

msh2 = msh1.clone().cut_with_mesh(cutmesh)
redcap = msh2.cap(return_cap=True).color("r4")

plt = Plotter(N=2, axes=1)
plt.at(0).show(msh1, cutmesh, __doc__)
plt.at(1).show(msh2, redcap, viewup="z")
plt.interactive().close()
