'''
Extracts the cells where scalar value satisfies a threshold criterion.
'''
from vtkplotter import Plotter, threshold, text

vp = Plotter(N=2, bg=(20,20,20), bg2='blackboard')

##################################### pointColors
man = vp.load('data/shapes/man.vtk')
scals = man.coordinates()[:,1] + 37 # pick y coords of vertices

man.pointColors(scals, cmap='cool') 
vp.show(man, at=0, viewup='z')
vp.addScalarBar(title='threshold', horizontal=True)

##################################### threshold
cutman = threshold(man, scals, vmin=36.9, vmax=37.5)

doc = text(__doc__, c='w')

vp.show([cutman, doc], at=1, interactive=1)

