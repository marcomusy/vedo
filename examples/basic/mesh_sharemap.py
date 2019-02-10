'''
Example on how to share the same a color map for different meshes.
'''
print(__doc__)
from vtkplotter import load


##################################### 
man1 = load('data/shapes/man.vtk')
scals = man1.coordinates()[:,2]*5 + 27 # pick z coordinates [18->34]

man1.pointColors(scals, cmap='jet', vmin=18, vmax=44) 
man1.show(N=2, at=0, axes=0, elevation=-80)

##################################### 
man2 = load('data/shapes/man.vtk').addScalarBar()
scals = man2.coordinates()[:,2]*5 + 37 # pick z coordinates [28->44]

man2.pointColors(scals, cmap='jet', vmin=18, vmax=44) 
man2.show(at=1, interactive=1)
