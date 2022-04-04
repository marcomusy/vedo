"""Modify mesh vertex positions"""
from vedo import *

t = Text2D(__doc__)

dsc = Disc(res=(8,120)).lineWidth(0.1)
coords = dsc.points()

for i in range(50):
    coords[:,2] = sin(i/10.*coords[:,0])/5 # move vertices in z
    dsc.points(coords)  # modify mesh
    show(dsc, t, resetcam=not i, interactive=0, axes=7) # resetcam only for i=0

interactive().close()
