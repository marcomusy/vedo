"""Set a jpeg background image on a
vtkRenderingWindow layer, after
the first rendering it can be
zoomed to fill the window."""
from vtkplotter import *

settings.showRendererFrame = False

doc = Text(__doc__, c="k", bg="w")

vp = Plotter(N=2, size=(400, 800), axes=4, sharecam=0,
			 bg=datadir+"images/tropical.jpg")

a1 = load(datadir+"flamingo.3ds").rotateX(-90)
a2 = Polygon()

vp.show(a1, doc, at=0)

vp.backgroundRenderer.GetActiveCamera().Zoom(2.5)

vp.show(a2, at=1, interactive=1)
