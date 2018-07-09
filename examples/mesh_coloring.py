# Example on how to specify a color for each individual cell 
# or point of an actor's mesh. Needs matplotlib.
from __future__ import division, print_function
import plotter


vp = plotter.vtkPlotter(shape=(1,3), axes=1)

man1 = vp.load('data/shapes/man.vtk')
Np = man1.N()
pscals = range(Np)
man1.pointScalars(pscals, 'mypointscalars')
#print(man1.scalars('mypointscalars')) # info can be retrieved this way
vp.show(man1, at=0)
vp.addScalarBar()  # add a scalarbar to last drawn actor


man2 = vp.load('data/shapes/man.vtk')
pscals = man2.coordinates()[:,1] + 37 # pick y coordinates of vertices
man2.pointColors(pscals, 'bone')      # use a colormap to associate a color
#print(man2.scalars('pointcolors_bone'))
vp.show(man2, at=1, legend='setPointColors')
vp.addScalarBar()


man3 = vp.load('data/shapes/man.vtk')
cscals = man3.cellCenters()[:,2] + 37 # pick z coordinates of cells
man3.cellColors(cscals, 'afmhot')
#print(man3.scalars('cellcolors_afmhot'))
txt = vp.text('floor temperature is 35C', pos=(-1,-.5,-1.7), s=.1)
vp.show([txt.rotateZ(90), man3], at=2, legend=['','setCellColors'])
vp.addScalarBar(man3) # specify the actor to which it must be added

vp.show(interactive=1)
