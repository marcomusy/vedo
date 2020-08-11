"""Make a icon to indicate orientation
and place it in one of the 4 corners
within the same renderer.
"""
from vedo import *


vp = Plotter(axes=5)
# axes type 5 builds an annotated orientation cube

vp.load(datadir+'porsche.ply').lighting('metallic')

vp.show(interactive=0)


elg = load(datadir+"images/embl_logo.jpg")
vp.addIcon(elg, pos=2, size=0.06)

vp.addIcon(VedoLogo(), pos=1, size=0.06)

vp += Text2D(__doc__, pos=8, s=0.8)

vp.show(interactive=1)
