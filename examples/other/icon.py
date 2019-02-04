'''
Make a icon actor to indicate orientation 
or for comparison and place it in one of 
the 4 corners within the same renderer.
'''
from vtkplotter import Plotter, text


vp = Plotter(axes=5, bg='white') 
# type 5 builds an annotated orientation cube

vp.load('data/270.vtk', c='blue', bc='v', legend=False)

vp.show(interactive=0)

vlg = vp.load('data/images/vtk_logo.png', alpha=.5)
vp.addIcon(vlg, pos=1)

elg = vp.load('data/images/embl_logo.jpg')
vp.addIcon(elg, pos=2, size=0.06)

a1 = vp.load('data/250.vtk', c=2)
a2 = vp.load('data/290.vtk', alpha=.4)
icon = vp.Assembly([a1, a2])
vp.addIcon(icon, pos=4) # 4=bottom-right

vp.add(text(__doc__, pos=8))

vp.show(interactive=1)
