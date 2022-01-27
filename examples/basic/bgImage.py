"""
Set a jpeg background image
on a separate rendering layer
"""
from vedo import *

plt = Plotter(N=4, sharecam=False,
			 bg=dataurl+"images/tropical.jpg",
			 bg2='light blue')

a1 = load(dataurl+"flamingo.3ds").rotateX(-90)

plt.show(__doc__, at=2)

# after first rendering, picture can be zoomed to fill the window:
plt.backgroundRenderer.GetActiveCamera().Zoom(1.8)

plt.show(VedoLogo(distance=2), at=0)

plt.show(a1, at=3).interactive().close()
