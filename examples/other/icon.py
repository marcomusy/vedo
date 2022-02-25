"""Make a icon to indicate orientation
and place it in one of the 4 corners
within the same renderer"""
from vedo import *

plt = Plotter(axes=5)

plt += Text3D(__doc__).bc('tomato')

elg = Picture(dataurl+"images/embl_logo.jpg")

plt.addIcon(elg, pos=2, size=0.06)
plt.addIcon(VedoLogo(), pos=1, size=0.06)

plt.show().close()
