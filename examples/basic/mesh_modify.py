"""Modify mesh vertex positions"""
from vedo import *

disc = Disc(res=(8,120)).linewidth(1)

plt = Plotter(interactive=False, axes=1)
plt.show(disc, Point(), __doc__)

for i in range(100):
    # Modify vertex positions
    disc.vertices += [0.01, 0.01*sin(i/20), 0]
    plt.reset_camera().render()

plt.interactive().close()
