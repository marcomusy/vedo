"""Hover mouse onto an object
to pop a flag-style label"""
from vedo import *

b = Mesh(dataurl+'bunny.obj').color('m')
c = Cube(side=0.1).compute_normals().alpha(0.8).y(-0.02)

fp = b.flagpole('A flag pole descriptor\nfor a rabbit', font='Quikhand')
fp.scale(0.5).color('v').use_bounds() # tell camera to take fp bounds into account

c.caption('2d caption for a cube\nwith face indices', point=[0.044, 0.03, -0.04],
          size=(0.3,0.06), font="VictorMono", alpha=1)

# create a new object made of polygonal text labels to indicate the cell numbers
labs = c.labels('id', on="cells", font='Theemim', scale=0.02, c='k')
# labs = c.labels2d(scale=3)

# create a custom entry to the legend
b.legend('Bugs the bunny')
c.legend('The Cube box')
lbox = LegendBox([b,c], font="Bongas", width=0.25)

show(b, c, fp, labs, lbox, __doc__, axes=11, bg2='linen').close()
