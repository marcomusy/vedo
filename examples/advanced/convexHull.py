"""Create the Convex Hull of a Mesh or a set of input points"""
from vedo import *

settings.defaultFont = 'Bongas'
settings.useDepthPeeling = True

spid = Mesh(dataurl+"spider.ply").c("brown")

ch = ConvexHull(spid.points()).alpha(0.2)

show(spid, ch, __doc__, axes=1).close()
