# Example on how to specify a color for each individual cell 
# or point of an actor's mesh. 
# Last example shows the usage of addScalarBar3D().
# Needs matplotlib.
from vtkplotter import Plotter


vp = Plotter(shape=(1,3), axes=1, size='fullscreen')


#####################################
man1 = vp.load('data/shapes/man.vtk')
Np = man1.N()
pscals = range(Np) # coloring will be by index nr of the vertex
man1.pointScalars(pscals, 'mypointscalars') # add a vtkArray to actor
#print(man1.scalars('mypointscalars')) # info can be retrieved this way
vp.show(man1, at=0)
vp.addScalarBar()  # add a scalarbar to last drawn actor


#####################################
man2 = vp.load('data/shapes/man.vtk')
pscals = man2.coordinates()[:,1] + 37    # pick y coordinates of vertices
man2.pointColors(pscals, 'bone')         # use a colormap to associate a color
#print(man2.scalars('pointcolors_bone')) # info can be retrieved this way
vp.show(man2, at=1, legend='setPointColors')
vp.addScalarBar(horizontal=True)


#####################################
man3 = vp.load('data/shapes/man.vtk')
cscals = man3.cellCenters()[:,2] + 37    # pick z coordinates of cells
man3.cellColors(cscals, 'afmhot')

# add some oriented 3D text
txt = vp.text('floor temperature is 35C', s=.1).rotateZ(90).pos([1,-.9,-1.7])

vp.show([txt, man3], at=2, legend=['','setCellColors'], axes=0)

# add a fancier 3D scalar bar
vp.addScalarBar3D(man3, at=2, pos=(-1,0,-1.7), cmap='afmhot')

vp.show(interactive=1, zoom=1.4)
