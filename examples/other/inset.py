"""Render meshes into inset windows
(which can be dragged)"""
from vedo import *

plt = Plotter(bg2='bisque', size=(1000,800), interactive=False)

e = Volume(dataurl+"embryo.tif").isosurface()
e.normalize().shift(-2,-1.5,-2).c("gold")

plt.show(e, __doc__, viewup='z')

# make clone copies of the embryo surface and cut them:
e1 = e.clone().cutWithPlane(normal=[0,1,0]).c("green")
e2 = e.clone().cutWithPlane(normal=[1,0,0]).c("red")

# add 2 draggable inset windows:
plt.addInset(e1, pos=(0.9,0.8))
plt.addInset(e2, pos=(0.9,0.5))

# customised axes can also be inserted:
ax = Axes(xrange=(0,1), yrange=(0,1), zrange=(0,1),
          xtitle='front', ytitle='left', ztitle='head',
          yzGrid=False, xTitleSize=0.15, yTitleSize=0.15, zTitleSize=0.15,
          xLabelSize=0, yLabelSize=0, zLabelSize=0, tipSize=0.05,
          axesLineWidth=2, xLineColor='dr', yLineColor='dg', zLineColor='db',
          xTitleOffset=0.05, yTitleOffset=0.05, zTitleOffset=0.05,
)

ex = e.clone().scale(0.25).pos(0,0.1,0.1).alpha(0.1).lighting('off')
plt.addInset(ax, ex, pos=(0.1,0.1), size=0.15, draggable=False)
plt.interactive().close()
