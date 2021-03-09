"""Make a icon to indicate orientation
and place it in one of the 4 corners
within the same renderer.
"""
from vedo import *


# axes type 5 builds an annotated orientation cube
vp = Plotter(axes=5)
vp.show(interactive=0)

elg = load(dataurl+"images/embl_logo.jpg")
vp.addIcon(elg, pos=2, size=0.06)

vp.addIcon(VedoLogo(), pos=1, size=0.06)

vp += Text3D(__doc__).bc('tomato')

vp.show(interactive=1)
