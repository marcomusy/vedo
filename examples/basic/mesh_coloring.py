"""
Example on how to specify a color for each individual cell 
or point of an actor's mesh. 
Last example also shows the usage of addScalarBar3D().
"""
print(__doc__)

from vtkplotter import *
import numpy as np

vp = Plotter(N=3)

##################################### addPointScalars
man1 = vp.load(datadir+"shapes/man.vtk")
Np = man1.N()  # nr. of vertices
scals = np.linspace(0, 1, Np)  # coloring by index nr of vertex

man1.addPointScalars(scals, "mypointscalars")  # add a vtkArray to actor
# print(man1.scalars('mypointscalars')) # info can be retrieved this way
vp.show(man1, at=0, elevation=-60)
vp.addScalarBar()  # add a scalarbar to last drawn actor


##################################### pointColors
man2 = vp.load(datadir+"shapes/man.vtk")
scals = man2.coordinates()[:, 1] + 37  # pick y coordinates of vertices

man2.pointColors(scals, cmap="bone", vmin=36.2, vmax=36.7)  # right dark arm
vp.show(man2, at=1, axes=0)
vp.addScalarBar(horizontal=True)


##################################### cellColors
man3 = vp.load(datadir+"shapes/man.vtk")
scals = man3.cellCenters()[:, 2] + 37  # pick z coordinates of cells
man3.cellColors(scals, cmap="afmhot")
# print(man3.scalars('cellColors_afmhot')) # info can be retrieved this way

# add some oriented 3D text
txt = Text("Floor temperature is 35C", pos=[1, -0.9, -1.7], s=0.1).rotateZ(90)
vp.show(man3, txt, at=2)

# add a fancier 3D scalar bar embedded in the scene
vp.addScalarBar3D(man3, at=2, pos=(-1, 0, -1.7))

vp.show(interactive=1)


# N.B. in the above example one can also do:
# import matplotlib.cm as cm
# man2.pointColors(scals, cmap=cm.bone)
