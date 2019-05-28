"""
Make a icon actor to indicate orientation 
and place it in one of the 4 corners 
within the same renderer.
"""
from vtkplotter import *


vp = Plotter(axes=5, bg="white")
# axes type 5 builds an annotated orientation cube

vp.load(datadir+'porsche.ply').lighting('metallic')

vp.show(interactive=0)

vlg = load(datadir+"images/vtk_logo.png", alpha=0.5)
vp.addIcon(vlg, pos=1)

elg = load(datadir+"images/embl_logo.jpg")
vp.addIcon(elg, pos=2, size=0.06)

plg = load(datadir+"images/vlogo_small.png")
vp.addIcon(plg, pos=4, size=0.1) # 4=bottom-right

vp += Text(__doc__, pos=8)

vp.show(interactive=1)
