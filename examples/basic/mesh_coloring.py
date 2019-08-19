"""
Example on how to specify a color for each individual cell
or point of an actor's mesh.
Last example also shows the usage of addScalarBar3D().
"""
print(__doc__)

from vtkplotter import *
import numpy as np

##################################### addPointScalars
man1 = load(datadir+"man.vtk")
nv = man1.N()  # nr. of vertices
scals = np.linspace(0, 1, nv)  # coloring by index nr of vertex

man1.addPointScalars(scals, "mypointscalars")  # add a vtkArray to actor
# print(man1.scalars('mypointscalars')) # info can be retrieved this way
man1.addScalarBar(c='white')  # add a default scalarbar
show(man1, at=0, N=3, axes=0, elevation=-60)


##################################### pointColors
man2 = load(datadir+"man.vtk")
scals = man2.getPoints()[:, 1] + 37  # pick y coordinates of vertices

man2.pointColors(scals, cmap="bone", vmin=36.2, vmax=36.7)  # right dark arm
man2.addScalarBar(horizontal=True)
show(man2, at=1)


##################################### cellColors
man3 = load(datadir+"man.vtk")
scals = man3.cellCenters()[:, 2] + 37  # pick z coordinates of cells
man3.cellColors(scals, cmap="afmhot")
# print(man3.scalars('cellColors_afmhot')) # info can be retrieved this way

# add some oriented 3D text
txt = Text("Floor temperature is 35C", pos=[1, -0.9, -1.7], s=0.1).rotateZ(90)

# add a fancier 3D scalar bar embedded in the scene
man3.addScalarBar3D(pos=(-1, 0, -1.7))
show(man3, txt, at=2, interactive=1)


# N.B. in the above example one can also do:
# import matplotlib.cm as cm
# man2.pointColors(scals, cmap=cm.bone)
