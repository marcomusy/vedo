'''
Make a icon actor to indicate orientation 
or for comparison and place it in one of 
the 4 corners within the same renderer.
'''
from vtkplotter import Plotter, load, Text


vp = Plotter(axes=5, bg='white') 
# type 5 builds an annotated orientation cube

a270 = load('data/270.vtk', c='blue', bc='v', legend=False)

vp.show(interactive=0)

vlg = load('data/images/vtk_logo.png', alpha=.5)
vp.addIcon(vlg, pos=1)

elg = load('data/images/embl_logo.jpg')
vp.addIcon(elg, pos=2, size=0.06)

a1 = load('data/250.vtk', c=2)
a2 = load('data/290.vtk', alpha=.4)
icon = a1 + a2          # vtkActor + vtkActor = vtkAssembly
vp.addIcon(icon, pos=4) # 4=bottom-right

vp.add(Text(__doc__, pos=8))

vp.show(a270, interactive=1)
