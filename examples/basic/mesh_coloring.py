"""Specify color mapping for cells
and points of a Mesh"""
from vedo import dataurl, Plotter, Mesh

plt = Plotter(N=3, axes=11)

##################################### add a cell array
man1 = Mesh(dataurl+"man_low.vtk").linewidth(0.1)
nv = man1.ncells                    # nr. of cells
scals = range(nv)                   # coloring by the index of cell
man1.cmap("Paired", scals, on='cells').add_scalarbar("cell nr")
plt.at(0).show(man1, __doc__, elevation=-60)


##################################### Point coloring
man2 = Mesh(dataurl+"man_low.vtk")
scals = man2.vertices[:, 0] + 37    # pick x coordinates of vertices
man2.cmap("hot", scals)
man2.add_scalarbar(horizontal=True)
plt.at(1).show(man2, "mesh.cmap()")


##################################### Cell coloring
man3 = Mesh(dataurl+"man_low.vtk")
scals = man3.cell_centers[:, 2] + 37  # pick z coordinates of cells
man3.cmap("afmhot", scals, on='cells')

# Add a fancier 3D scalar bar embedded in the 3d scene
man3.add_scalarbar3d(size=[0.2,3])
man3.scalarbar.scale(1.1).rotate_x(90).shift([0,2,0])

plt.at(2).show(man3, "mesh.cmap(on='cells')")
plt.interactive()
plt.close()
