"""Modify mesh vertex positions"""
from vedo import *

dsc = Disc(res=(8,120)).linewidth(0.1)

plt = Plotter(interactive=False, axes=7)
plt.show(dsc, __doc__)

coords = dsc.points()
for i in range(100):
    coords[:,2] = sin(i/10.*coords[:,0])/5 # move vertices in z
    dsc.points(coords)  # update mesh points
    plt.render()

plt.interactive().close()
