"""Renderer meshes into insets
(can be dragged)
"""
from vedo import *

vp = Plotter(axes=1)

e = load(datadir+"embryo.tif").isosurface()
e.normalize().c("gold")

vp.show(e, __doc__, viewup='z', interactive=0)

e1 = e.clone().cutWithPlane(normal=[1,0,0]).c("red")
e2 = e.clone().cutWithPlane(normal=[0,1,0]).c("pink")
e3 = e.clone().cutWithPlane(normal=[0,0,1]).c("blue")

vp.showInset(e1, pos=(0.9,0.2))
vp.showInset(e2, pos=(0.9,0.5))
vp.showInset(e2, e3, pos=(0.9,0.8))

vp.show(interactive=1)
