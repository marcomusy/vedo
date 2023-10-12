"""Create the Convex Hull of a Mesh or a set of input points"""
from vedo import *

settings.default_font = 'Bongas'
settings.use_depth_peeling = True

spid = Mesh(dataurl+"spider.ply").c("brown")

ch = ConvexHull(spid.vertices).alpha(0.2)

show(spid, ch, __doc__, axes=1).close()
