"""
Set a jpeg background image
on a separate rendering layer
"""
from vtkplotter import *

settings.showRendererFrame = False

vp = Plotter(N=4, sharecam=False,
			 bg=datadir+"images/harvest.jpg",
			 bg2='light blue')

doc = Text2D(__doc__, c="w", bg="w")
a1 = load(datadir+"flamingo.3ds").rotateX(-90)

vp.show(doc, at=0)
# after first rendering, picture can be zoomed to fill the window:
vp.backgroundRenderer.GetActiveCamera().Zoom(2.0)

vp.show(a1, at=3, interactive=1)
