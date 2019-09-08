"""
Set a jpeg background image
on a separate rendering layer
"""
from vtkplotter import *

settings.showRendererFrame = False

vp = Plotter(N=9, sharecam=False,
			 bg=datadir+"images/tropical.jpg",
			 bg2='light blue')

doc = Text(__doc__, c="k", bg="w")
a1 = load(datadir+"flamingo.3ds").rotateX(-90)
a2 = load(datadir+"grasshop.wrl").rotateY(-90)
logo = load(datadir+"images/vlogo_large.png").alpha(0.3).rotateY(-20)

vp.show(doc, at=0)
# after first rendering, picture can be zoomed to fill the window:
vp.backgroundRenderer.GetActiveCamera().Zoom(2.0)

vp.show(logo, at=2)
vp.show(a1, at=6)
vp.show(a2, at=5, interactive=1)
