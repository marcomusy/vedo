# Make a icon actor to indicate orientation or for comparison
# and place it in one of the 4 corners within the same renderer
#
from vtkplotter import Plotter


vp = Plotter(axes=5) # type 5 builds an annotated orientation cube

act = vp.load('data/270.vtk', c='blue', bc='v')

vp.render()

a1 = vp.load('data/250.vtk', c=2)
a2 = vp.load('data/290.vtk', alpha=.4)
icon = vp.Assembly([a1, a2])
vp.addIcon(icon, pos=4) # 4=bottom-right

lg = vp.load('data/images/embl_logo.jpg')
vp.addIcon(lg, pos=1)

vp.show(act)
