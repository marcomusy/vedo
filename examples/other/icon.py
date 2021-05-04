"""Make a icon to indicate orientation
and place it in one of the 4 corners
within the same renderer.
"""
from vedo import *


# axes type 5 builds an annotated orientation cube
plt = Plotter(axes=5)
plt.show(interactive=0)

elg = load(dataurl+"images/embl_logo.jpg")
plt.addIcon(elg, pos=2, size=0.06)

plt.addIcon(VedoLogo(), pos=1, size=0.06)

plt += Text3D(__doc__).bc('tomato')

plt.show(interactive=True).close()
