"""Create the Convex Hull of
a Mesh or a set of input points
"""
from vedo import *

spid = load(datadir+"spider.ply").c("brown")

ch = convexHull(spid.points()).alpha(0.2)

show(spid, ch, __doc__, axes=1)
