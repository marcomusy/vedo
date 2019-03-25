"""
Make a icon actor to indicate orientation 
or for comparison and place it in one of 
the 4 corners within the same renderer.
"""
from vtkplotter import Plotter, load, Text, datadir


vp = Plotter(axes=5, bg="white")
# type 5 builds an annotated orientation cube

a270 = load(datadir+"270.vtk", c="blue", bc="v")

vp.show(interactive=0)

vlg = load(datadir+"images/vtk_logo.png", alpha=0.5)
vp.addIcon(vlg, pos=1)

elg = load(datadir+"images/embl_logo.jpg")
vp.addIcon(elg, pos=2, size=0.06)

a1 = load(datadir+"250.vtk", c=2)
a2 = load(datadir+"290.vtk", alpha=0.4)
icon = a1 + a2  # vtkActor + vtkActor = vtkAssembly
vp.addIcon(icon, pos=4)  # 4=bottom-right

vp.add(Text(__doc__, pos=8))

vp.show(a270, interactive=1)
