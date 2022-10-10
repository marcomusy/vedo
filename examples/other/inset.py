"""Render meshes into inset windows
(which can be dragged)"""
from vedo import *

plt = Plotter(bg2='bisque', size=(1000,800), interactive=False)

e = Volume(dataurl+"embryo.tif").isosurface()
e.normalize().shift(-2,-1.5,-2).c("gold")

plt.show(e, __doc__, viewup='z')

# make clone copies of the embryo surface and cut them:
e1 = e.clone().cut_with_plane(normal=[0,1,0]).c("green4")
e2 = e.clone().cut_with_plane(normal=[1,0,0]).c("red5")

# add 2 draggable inset windows:
plt.add_inset(e1, pos=(0.9,0.8))
plt.add_inset(e2, pos=(0.9,0.5))

# customised axes can also be inserted:
ax = Axes(
    xrange=(0,1), yrange=(0,1), zrange=(0,1),
    xtitle='front', ytitle='left', ztitle='head',
    yzgrid=False, xtitle_size=0.15, ytitle_size=0.15, ztitle_size=0.15,
    xlabel_size=0, ylabel_size=0, zlabel_size=0, tip_size=0.05,
    axes_linewidth=2, xline_color='dr', yline_color='dg', zline_color='db',
    xtitle_offset=0.05, ytitle_offset=0.05, ztitle_offset=0.05,
)

ex = e.clone().scale(0.25).pos(0,0.1,0.1).alpha(0.1).lighting('off')
plt.add_inset(ax, ex, pos=(0.1,0.1), size=0.15, draggable=False)
plt.interactive().close()
