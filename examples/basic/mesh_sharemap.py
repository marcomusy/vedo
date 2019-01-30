'''
Example on how to share the same a color map for different meshes.
'''
print(__doc__)
from vtkplotter import Plotter

vp = Plotter(N=2)

##################################### 
man1 = vp.load('data/shapes/man.vtk')
scals = man1.coordinates()[:,2]*5 + 37 # pick z coordinates [28->44]

man1.pointColors(scals, cmap='jet', vmin=18, vmax=44) 
vp.show(man1, at=0, axes=0, elevation=-80)

##################################### 
man2 = vp.load('data/shapes/man.vtk')
scals = man2.coordinates()[:,2]*5 + 27 # pick z coordinates [18->34]

man2.pointColors(scals, cmap='jet', vmin=18, vmax=44) 
vp.show(man2, at=1, axes=0)

vp.addScalarBar()
vp.show(interactive=1)

